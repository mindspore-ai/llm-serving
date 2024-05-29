import time
import math
from typing import List, Tuple, Deque
import logging

from queue import Queue
import copy
from mindspore_serving.serving_utils.constant import *
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus, EntryData
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.schedule.cache_engine import ServingBlockMemPool, ServingCacheEngine

class FastserveMulQue:
    """fastserve multil waitting request queue"""
    def __init__(self, time):
        self.waiting_request_queue: Deque[EntryMetaData] = Deque([])
        self.time = time

class Priority(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

class Schedule:
    """static batch strategy"""

    def __init__(self, config: ServingConfig):
        self.waiting_request_queue: Deque[EntryMetaData] = Deque([])
        self.running_request_list: List[EntryMetaData] = []
        self.count_of_invalid_sample = 0
        self.config = config
        self.batch_size = config.model_config.decode_batch_size[0]
        self.eos_token = config.model_config.end_token
        self.batch_waiting_time = config.serving_config.prefill_batch_waiting_time
        self.decode_batch_waiting_time = config.serving_config.decode_batch_waiting_time
        self.batching_strategy = config.model_config.batching_strategy
        self.max_input_len = config.model_config.seq_length[-1] if len(config.model_config.seq_length) > 0 else 4096
        # batch中有效token的最大index, 初始化为-1
        self.max_valid_index = -1
        self.dyn_batch = config.model_config.decode_batch_size
        # for fastserve batching
        if config.model_config.fastserve:
            self.exec_time_slice = config.fa_config.time_slice
            # 最高优先级等待队列的执行时间片，后面每个次优先级等待队列的执行时间片是之前的rate倍
            # 总共5个优先级等待队列 ：
            # self.waiting_request_queue是最高优先级等待队列
            # self.queue_list存放6个次优先级等待队列
            # 但其中优先级最低的2个队列只用于接收被反复降级的请求，保证其得到较长时间片而不会被反复抢占、降级
            self.queue_list = [FastserveMulQue(self.exec_time_slice*(config.fa_config.rate**(i+1))) for i in range(6)]
            self.waiting_time = config.fa_config.starve_waiting_time
            self.starve_times = {}
            self.index_priority = {}
            self.entry_in_host = []
            self.start_times = {}
            self.new_insert_entry = set()


    def get_dyn_batch(self):
        return self.batch_size

    def get_queue_len(self):
        return len(self.waiting_request_queue)

    def add_entrys(self, entry_meta_data: EntryMetaData):
        entry_meta_data.get_entry_data().set_status(EntryStatus.WAITING)
        self.waiting_request_queue.append(entry_meta_data)

    def cal_enset_priority(self, prompt_len):
        priority = math.ceil(math.log2(prompt_len / 10))
        if priority < 0:
            priority = 0
        return Priority(min(priority, 4))

    def add_entrys_fa(self, entry_meta_data: EntryMetaData):
        entry_meta_data.get_entry_data().set_status(EntryStatus.WAITING)
        prompt_len = entry_meta_data.entry_data.get_prompt_len()
        priority = self.cal_enset_priority(prompt_len)
        if priority ==  Priority.ZERO:
            self.waiting_request_queue.append(entry_meta_data)
        else:
            self.queue_list[priority.value - 1].waiting_request_queue.append(entry_meta_data)
        self.starve_times[entry_meta_data.request_id] = time.time()
        self.index_priority[entry_meta_data.request_id] = priority.value
        self.start_times[entry_meta_data.request_id] = time.time()

    def _padding_batch_size(self):
        while len(self.running_request_list) < self.batch_size:
            entry_meta_data = copy.deepcopy(self.running_request_list[-1])
            entry_meta_data.entry_data.set_status(EntryStatus.PADDING_INVAILED)
            self.running_request_list.append(entry_meta_data)

    def _over_all_complete_entry(self):
        for index, _ in enumerate(self.running_request_list):
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)

    def _padding_request_into_batching_list(self, index):
        if not self.waiting_request_queue:
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            if not self.waiting_request_queue:
                entry_meta_data = copy.deepcopy(self.running_request_list[-1])

                if entry_meta_data.entry_data.get_prompt_len() + entry_meta_data.entry_data.get_max_token_len() >= self.max_input_len:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)

                entry_meta_data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(entry_meta_data)
                logging.debug(f'waiting and add invalid request in batch init, batch size index is {index}')
            else:
                data = self.waiting_request_queue.popleft()
                if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)

                data.get_entry_data().set_decode_index(index)
                self.running_request_list.append(data)
                logging.debug(f'add new valid request in batch, batch size index is {index}')
        else:
            data = self.waiting_request_queue.popleft()
            logging.debug('get_nowait2')

            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(index)
            self.running_request_list.append(data)
            logging.debug(f'add new valid request in batch, batch size index is {index}')

    def _get_next_batch(self):
        self.running_request_list.clear()
        count = 0
        # no request in schedule queue, return
        if not self.waiting_request_queue:
            return
        # add request into batching list
        while self.waiting_request_queue:
            if count >= self.batch_size:
                break
            data = self.waiting_request_queue.popleft()

            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)

            data.get_entry_data().set_decode_index(count)
            self.running_request_list.append(data)
            logging.debug(f'add new valid request in batch, batch size index is {count}')
            count += 1
        # if batching list not full, add invalid padding request into batching list
        if len(self.running_request_list) < self.batch_size + 1:
            for index in range(len(self.running_request_list), self.batch_size):
                self._padding_request_into_batching_list(index)

    def _all_samples_in_batch_is_over(self) -> bool:
        res = True
        for _, data in enumerate(self.running_request_list):
            if data.get_entry_data().get_status() == EntryStatus.RUNNING:
                res = False
        return res

    def checkout_entry(self) -> List[bool]:
        """
          request in FINISHED_LENGTH_CAPPED, FINISHED_STOPPED, PADDING_INVAILED status can be cut out
        """
        checkout_list = []
        for index, data in enumerate(self.running_request_list):
            check_ = False
            # max_length, cut out finished request in batch
            if data.get_entry_data().get_status() == EntryStatus.FINISHED_LENGTH_CAPPED:
                check_ = True
            # eos, cut out finished request in batch
            elif data.get_entry_data().get_status() == EntryStatus.FINISHED_STOPPED:
                check_ = True
            elif data.get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                check_ = True
            checkout_list.append(check_)
        return checkout_list

    def _padding_new_prompt_to_batch(self, index):
        # queue is empty, no new request in schedule queue
        if not self.waiting_request_queue:
            # waiting
            time.sleep(self.batch_waiting_time / float(len(self.running_request_list)))
            # no new request, continue finished valid decode
            if not self.waiting_request_queue:
                # logging.debug('waiting and no new request, continue finished valid decode')
                return
            # new requestes in queue
            else:
                # logging.debug('add a new request into batching list')
                data = self.waiting_request_queue.popleft()
                # logging.debug('get_nowait3')
                if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                    data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)

                else:
                    data.get_entry_data().set_status(EntryStatus.RUNNING)
                data.get_entry_data().set_decode_index(index)
                self.running_request_list[index] = data
                # logging.debug(f'add new valid request in batch, batch size index is {index}')
        else:
            # logging.debug('add a new request into batching list')
            data = self.waiting_request_queue.popleft()
            # logging.debug('get_nowait4')
            if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
                data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
            else:
                data.get_entry_data().set_status(EntryStatus.RUNNING)
            data.get_entry_data().set_decode_index(index)
            self.running_request_list[index] = data
            # logging.debug(f'add new valid request in batch, batch size index is {index}')

    def _update_status_after_one_itreation(self):
        self.count_of_invalid_sample = 0
        """checkout and update number of invalid request in batching list"""
        self.max_valid_index = -1
        for index, data in enumerate(self.running_request_list):
            data_status = data.get_entry_data().get_status()
            if data_status == EntryStatus.FINISHED_STOPPED or data_status == EntryStatus.FINISHED_LENGTH_CAPPED:
                self.count_of_invalid_sample += 1
            elif data_status == EntryStatus.RUNNING:
                self.max_valid_index = index

    def _determine_batch_size(self):
        self._update_status_after_one_itreation()
        bf_batch = self.batch_size
        queue_len = len(self.waiting_request_queue)
        bs_list_len = len(self.dyn_batch)
        # 1. 请求队列长度大于当前batch_size，扩容
        if self.max_valid_index == -1 or queue_len > self.batch_size:
            # 获取最接近waiting list长度的batch档位
            dyn_index = queue_len
        # 2. 请求队列长度小于count_of_invalid_sample，根据max_valid_index动态到邻近档位
        elif queue_len < self.count_of_invalid_sample:
            # max_valid_index左侧有多少结束的token
            left_free_num = self.count_of_invalid_sample - (self.batch_size - self.max_valid_index - 1)
            if queue_len <= left_free_num:
                dyn_index = self.max_valid_index + 1
            else:
                # 请求队列中全部补齐会到哪个index
                dyn_index = queue_len - left_free_num + self.max_valid_index + 1
        else:
            dyn_index = self.max_valid_index + 1 + queue_len - self.count_of_invalid_sample
        bs_after_changing = self.batch_size
        if dyn_index <= 0:
            # 默认值
            bs_after_changing = self.dyn_batch[0]
        else:
            for i in range(1, bs_list_len):
                if dyn_index > self.dyn_batch[bs_list_len - i - 1]:
                    bs_after_changing = self.dyn_batch[bs_list_len - i]
                    break
        self.batch_size = bs_after_changing if bs_after_changing > 0 else self.dyn_batch[0]
        af_batch = self.batch_size
        if af_batch != bf_batch:
            logging.debug(('----bs changed from  {} '.format(bf_batch)))
            logging.debug(('----bs changed to  {} '.format(af_batch)))
        if bf_batch >= af_batch:
            self.running_request_list = self.running_request_list[:af_batch]
        else:
            bf_batch = 0 if self.max_valid_index == -1 else bf_batch
            block_size = 0
            if self.config.model_config.page_attention:
                block_size = self.config.pa_config.block_size
            for i in range(bf_batch, af_batch):
                entry_meta_data = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                                request_id=PADDING_REQUEST_ID,
                                                is_prompt=True,
                                                entry_data=EntryData(prompt_tokens=[self.eos_token],
                                                                     max_token_len=5000),
                                                entry_id=-1,
                                                prompt=PADDING_PROMPT,
                                                block_size=block_size)
                entry_meta_data.get_entry_data().set_decode_index(i)
                entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
                self.running_request_list.append(entry_meta_data)

    def _continuous_batch(self):
        # init batch size when running_request_list is empty.
        if len(self.running_request_list) == 0:
            self._get_next_batch()
        # update invalid request number in batching list.
        self._update_status_after_one_itreation()
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()
        # update status after one inference step
        else:
            checkout_list = self.checkout_entry()
            for index, data in enumerate(checkout_list):
                if data and index < self.batch_size:
                    # logging.debug('----{}-th prefill request in batching padded to batch.'.format(index))
                    self._padding_new_prompt_to_batch(index)

    def _insert_new_prompt_to_batch_pa(self, index):
        # logging.debug('add a new request into batching list')
        # data = self.waiting_request_queue.get_nowait()
        data = self.waiting_request_queue.popleft()
        if data.entry_data.get_prompt_len() + data.entry_data.get_max_token_len() >= self.max_input_len:
            data.get_entry_data().set_status(EntryStatus.INPUT_OUTOFRANGE)
        else:
            data.get_entry_data().set_status(EntryStatus.RUNNING)
        data.get_entry_data().set_decode_index(index)
        self.running_request_list[index] = data
        if self.config.model_config.fastserve:
            self.new_insert_entry.add(data.request_id)
        logging.debug(f'add new valid request in batch, batch size index is {index}')

    def try_substitute_entry(self):
        checkout_list = self.checkout_entry()
        is_invalid_index_list = []
        for index, is_invalid in enumerate(checkout_list):
            if is_invalid:
                is_invalid_index_list.append(index)
        if not is_invalid_index_list:
            # logging.debug("no invalid entry to substitute")
            return False
        # 如果有空槽位，尝试替代一条新请求：
        index_to_substitute = is_invalid_index_list[0]
        logging.debug("trying to substitute old entry at index: %s", index_to_substitute)
        # 如果新entry需要的block数量，小于can_substitute entry的block数量 + mem pool全局剩余block数量
        new_entry = self.waiting_request_queue[0]
        if new_entry.cache_engine.try_use_budget(new_entry.get_entry_data().get_len()):
            self._insert_new_prompt_to_batch_pa(index_to_substitute)
            return True
        logging.debug("failed inserting to existing entry")
        # 如果空间不足，那么连第一条waiting的请求就无法替换，直接退出
        return False

    def reset_all_budgets(self):
        # logging.debug("current running list")
        for entry in self.running_request_list:
            entry.cache_engine.release_budget()

    def can_predict_current_batch(self):
        checkout_list = self.checkout_entry()
        for index, is_invalid in enumerate(checkout_list):
            # 对于batch中running的请求
            if is_invalid:
                continue
            entry = self.running_request_list[index]
            entry_cache_engine = entry.cache_engine
            if entry.is_prompt:
                if not entry_cache_engine.try_use_budget(entry.get_entry_data().get_len()):
                    return False
            else:
                if not entry_cache_engine.try_use_budget(1):
                    return False
        # logging.debug("can decode current batch return true")
        return True

    def try_initialize_paddings_pa(self):
        # running list和batch size不匹配时，添加padding位补充
        # 场景：1.启动server后，runninglist为空；2.升档后
        # logging.debug("try initialize paddings...")
        if len(self.running_request_list) == self.batch_size:
            return
        elif len(self.running_request_list) > self.batch_size:
            raise RuntimeError("running list size: %s larger than batch size: %s!", len(self.running_request_list),
                               self.batch_size)
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        for index in range(len(self.running_request_list), self.batch_size):
            padding_entry = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                          request_id=PADDING_REQUEST_ID,
                                          is_prompt=False,  # True
                                          entry_data=EntryData(prompt_tokens=[self.config.model_config.end_token],
                                                               max_token_len=5000),
                                          entry_id=-1,
                                          prompt=PADDING_PROMPT,
                                          block_size=block_size)
            padding_entry.get_entry_data().set_decode_index(index)
            padding_entry.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
            cache_engine = padding_entry.cache_engine
            cache_engine.assign_null_block()
            self.running_request_list.append(padding_entry)

    def insert_padding_entry(self, index):
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        padding_entry = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                      request_id=PADDING_REQUEST_ID,
                                      is_prompt=False,  # True
                                      entry_data=EntryData(prompt_tokens=[self.config.model_config.end_token],
                                                           max_token_len=5000),
                                      entry_id=-1,
                                      prompt=PADDING_PROMPT,
                                      block_size=block_size)
        padding_entry.get_entry_data().set_decode_index(index)
        padding_entry.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
        cache_engine = padding_entry.cache_engine
        cache_engine.assign_null_block()
        self.running_request_list[index] = padding_entry

    def try_swap_valid_entries(self):
        is_invalid_list = self.checkout_entry()
        num_tokens_index_list = []
        for index, is_invalid in enumerate(is_invalid_list):
            if is_invalid:
                continue
            num_tokens_index_list.append((self.running_request_list[index].get_entry_data().get_len(), index))
        if not num_tokens_index_list:
            raise RuntimeError("no valid entry to pop!")

        num_tokens_index_list.sort(key=lambda x: x[0])
        _, index_to_swap = num_tokens_index_list[0]

        # 释放一条长度最短的valid entries（认为是最后进来的，TODO：按照时间顺序pop掉最晚进来的entry）
        entry_to_swap = self.running_request_list[index_to_swap]
        entry_to_swap.get_entry_data().set_status(EntryStatus.WAITING)
        entry_to_swap.get_entry_data().set_decode_index(0)
        entry_to_swap.is_prompt = True
        entry_to_swap.cache_engine.release_cache()
        # append回waiting list
        logging.warning("swap entry out, index: %s", index_to_swap)
        self.waiting_request_queue.appendleft(entry_to_swap)
        # 用padding替代
        # logging.debug("inserting padding to popped entry %s", index_to_swap)
        self.insert_padding_entry(index_to_swap)

    def check_entry_in_host(self):
        now = time.time()
        for entry, enst in self.entry_in_host:
            if now - self.starve_times[entry.request_id] + enst >= self.waiting_time:
                if entry.cache_engine.try_use_budget(entry.get_entry_data().get_len()):
                    self.waiting_request_queue.appendleft(entry)
                    self.starve_times[entry.request_id] = time.time()
                    self.index_priority[entry.request_id] = 0

    #后续需修改，对max_enst_entry进行swap或release
    def find_max_enst_entry(self):
        max_enst_entry = [None for _ in range(2)]
        max_enst = [0 for _ in range(2)]

        for data in self.waiting_request_queue:
            enst = self.cal_enst(data)
            if enst > max_enst[0]:
                max_enst[0] = enst
                max_enst_entry[0] = data

        que_index = 0  
        for index, que in enumerate(self.queue_list):
            for data in que.waiting_request_queue:
                enst = self.cal_enst(data)
                if enst > max_enst[1]:
                    max_enst[1] = enst
                    max_enst_entry[1] = data
                    que_index = index
        
        flag = 1 if max_enst[0] <= max_enst[1] else 0


        if flag == 0 and max_enst_entry[0] != None:
            max_enst_entry[0].get_entry_data().set_status(EntryStatus.WAITING)
            max_enst_entry[0].is_prompt = True
            max_enst_entry[0].cache_engine.release_cache()
            self.starve_times[max_enst_entry[0].request_id] = time.time()
            self.entry_in_host.append((max_enst_entry[0],max_enst[0]))
            self.index_priority.pop(max_enst_entry[0].request_id, None)
            count = 0
            que_len = len(self.waiting_request_queue)
            while len(self.waiting_request_queue) != 0:
                data = self.waiting_request_queue.popleft()
                if data.request_id != max_enst_entry[0].request_id:
                    self.waiting_request_queue.append(data)
                else:
                    break
                count += 1
                if count == que_len:
                    break
        elif flag == 1 and max_enst_entry[1] != None:
            max_enst_entry[1].get_entry_data().set_status(EntryStatus.WAITING)
            max_enst_entry[1].is_prompt = True
            max_enst_entry[1].cache_engine.release_cache()
            self.starve_times[max_enst_entry[1].request_id] = time.time()
            self.entry_in_host.append((max_enst_entry[1], max_enst[1]))
            self.index_priority.pop(max_enst_entry[1].request_id, None)
            count = 0
            que_len = len(self.queue_list[que_index].waiting_request_queue)
            while len(self.queue_list[que_index].waiting_request_queue) != 0:
                data = self.queue_list[que_index].waiting_request_queue.popleft()
                if data.request_id != max_enst_entry[1].request_id:
                    self.queue_list[que_index].waiting_request_queue.append(data)
                else:
                    break
                count += 1
                if count == que_len:
                    break

    #往最高优先级等待队列补充一条请求，如果最高优先级等待队列为空的话
    def _refresh_queue_list(self):
        #优先处理饥饿请求
        self.promote_starved_jobs()

        #补充最高优先级队列的请求
        if len(self.waiting_request_queue) >= 1:
            return 
        padding_size = 1
        if padding_size > 0:
            count = 0
            for index, ele in enumerate(self.queue_list):
                while len(ele.waiting_request_queue) != 0 and count < padding_size:
                    data = ele.waiting_request_queue.popleft()
                    self.waiting_request_queue.append(data)
                    count += 1
                if count == padding_size:
                    break

    def try_substitute_entry_fastserve(self):
        checkout_list = []
        for index, data in enumerate(self.running_request_list):
            # 0 for FINISHED_LENGTH_CAPPED、FINISHED_STOPPED、PADDING_INVAILED, 
            # 1 for RUNNING
            # 2 for else
            check_ = 2
            if data.get_entry_data().get_status() == EntryStatus.RUNNING:
                index = self.index_priority.get(data.request_id)
                if index == None:
                    index = self.cal_enset_priority(data.entry_data.get_prompt_len())
                now = time.time()
                if index == 0 and now - self.start_times[data.request_id] > self.exec_time_slice:
                    check_ = 1
                if index != 0 and now - self.start_times[data.request_id] > self.queue_list[index-1].time:
                    check_ = 1
            elif data.get_entry_data().get_status() == EntryStatus.FINISHED_LENGTH_CAPPED:
                check_ = 0
                self.starve_times.pop(data.request_id, None)
                self.index_priority.pop(data.request_id, None)
                self.start_times.pop(data.request_id, None)
            elif data.get_entry_data().get_status() == EntryStatus.FINISHED_STOPPED:
                check_ = 0
                self.starve_times.pop(data.request_id, None)
                self.index_priority.pop(data.request_id, None)
                self.start_times.pop(data.request_id, None)
            elif data.get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                check_ = 0
                self.starve_times.pop(data.request_id, None)
                self.index_priority.pop(data.request_id, None)
                self.start_times.pop(data.request_id, None)
            checkout_list.append(check_)
        flag = True
        swap_index_list = []
        for index, check in enumerate(checkout_list):
            if check == 0 or check == 1:
                flag = False
                swap_index_list.append((index, check))
        if flag:
            return False
        # 如果有空槽位，尝试替代一条新请求：
        swap_index_list = sorted(swap_index_list, key=lambda x: x[1])
        index_to_substitute = swap_index_list[0][0]
        self._refresh_queue_list()
        new_entry = self.waiting_request_queue[0]
        flag_try_use_budget = False
        if new_entry.cache_engine.try_use_budget(new_entry.get_entry_data().get_len() if new_entry.is_prompt == True else 1):
            flag_try_use_budget = True
        if flag_try_use_budget:
            #如果swap的是非法请求
            if swap_index_list[0][1] == 0:
                self._insert_new_prompt_to_batch_pa(index_to_substitute)
                return True
            #如果swap的是running请求
            else:
                if self.update_entry_status(self.running_request_list[index_to_substitute]):
                    self._insert_new_prompt_to_batch_pa(index_to_substitute)
                    return True
                else:
                    logging.debug("update失败，无法抢占请求")
                    return False
        # 如果空间不足，那么连第一条waiting的请求就无法替换，直接退出
        return False

    # Promote starved jobs
    def promote_starved_jobs(self):
        for key, value in self.starve_times.items():
            if time.time() - value > self.waiting_time:
                index = self.index_priority[key]
                if index == 0:
                    continue
                else:
                    count = 0
                    que_len = len(self.queue_list[index-1].waiting_request_queue)
                    while len(self.queue_list[index-1].waiting_request_queue) != 0:
                        data = self.queue_list[index-1].waiting_request_queue.popleft()
                        if data.request_id != key:
                            self.queue_list[index-1].waiting_request_queue.append(data)
                        else:
                            self.starve_times[data.request_id] = time.time()
                            self.waiting_request_queue.append(data)
                            break
                        count += 1
                        if count == que_len:
                            break
                    
                    
    
    # 对执行队列中的request决定是否降低优先级，重新设置优先级，暂时默认降低一级, 需修改
    def update_entry_status(self, entry_meta_data: EntryMetaData):
        if entry_meta_data == None:
            return False
        index_pre = self.index_priority.get(entry_meta_data.request_id)
        if index_pre == None:
            index_pre = self.cal_enset_priority(entry_meta_data.get_entry_data.get_prompt_len())
        self.start_times[entry_meta_data.request_id] = time.time()
        index_next = index_pre + 1
        if index_next >= len(self.queue_list):
            index_next = len(self.queue_list) - 1
        self.index_priority[entry_meta_data.request_id] = index_next
        entry_meta_data.get_entry_data().set_status(EntryStatus.RUNNING)
        entry_meta_data.get_entry_data().set_decode_index(0)
        self.queue_list[index_next-1].waiting_request_queue.appendleft(entry_meta_data)
        self.starve_times[entry_meta_data.request_id] = time.time()
        return True

    #单个请求从等待队列pre到next需要的时间
    def exec_time(self, pre, next):
        res = 0
        for i in range(pre, next):
            res += self.queue_list[pre].time
        return res


    def cal_enst(self, entry_meta_data: EntryMetaData):
        priority = self.index_priority[entry_meta_data.request_id]
        exec_time = 0
        # # #最高优先级 priority = 0
        # # #第二优先级 priority = 1, 在self.queue_list中的index为 priority - 1
        
        if priority > 0:
            exec_time += len(self.waiting_request_queue) * 1
            exec_time += len(self.waiting_request_queue) * self.exec_time(0, priority-1)
            for i in range(0, priority-1):
                exec_time += len(self.queue_list[i].waiting_request_queue) * self.exec_time(i, priority-1)
        # #饥饿调度剩余时间
        now = time.time()
        near_starve_time = 0 if now - self.starve_times[entry_meta_data.request_id] > self.waiting_time else self.waiting_time - (now - self.starve_times[entry_meta_data.request_id])
        return min(exec_time, near_starve_time)


    def _continuous_batch_fastserve(self):
        start_time = time.time()
        #每次迭代之前，判断内存里的entry是否会在本次迭代执行，如果是，放回NPU里
        self.check_entry_in_host()

        ServingBlockMemPool.instance().reset_budget()
        # ServingBlockMemPool.instance().log_status()
        self.try_initialize_paddings_pa()
        # 判断batch内的running entry，能否进行本轮推理？
        num_entry_swapped_out = 0
        while not self.can_predict_current_batch():
            # 如果不能，swap出去已有请求
            self.reset_all_budgets()
            self.find_max_enst_entry()
            num_entry_swapped_out += 1
        if num_entry_swapped_out:
            self.reset_all_budgets()
            return
        # 3. 处理新请求
        self._refresh_queue_list()
        self.new_insert_entry.clear()
        while self.waiting_request_queue:
            # 如果有空batch槽，尝试插入
            if not self.try_substitute_entry_fastserve():
                # 尝试失败，退出
                break
            self._refresh_queue_list()

        self.reset_all_budgets()

        end_time = time.time()
        iter_time = end_time - start_time
        for key, value in self.start_times.items():
            self.start_times[key] = value + iter_time
        for id in self.new_insert_entry:
            self.start_times[id] = end_time

    def _continuous_batch_pa(self):
        ServingBlockMemPool.instance().reset_budget()
        # ServingBlockMemPool.instance().log_status()
        self.try_initialize_paddings_pa()
        # self.log_running_list("schedule start running status")
        # 判断batch内的running entry，能否进行本轮推理？
        num_entry_swapped_out = 0
        while not self.can_predict_current_batch():
            # 如果不能，swap出去已有请求
            self.reset_all_budgets()
            self.try_swap_valid_entries()
            num_entry_swapped_out += 1
        if num_entry_swapped_out:
            self.reset_all_budgets()
            return
        # 3. 处理新请求
        # logging.debug("determine if can process new request...")
        while self.waiting_request_queue:
            # 如果有空batch槽，尝试插入
            # logging.debug("has new entry, trying to enter current batch")
            if not self.try_substitute_entry():
                # 尝试失败，退出
                break
        self.reset_all_budgets()
        # ServingBlockMemPool.instance().log_status()

    def _static_batch(self):
        if self._all_samples_in_batch_is_over() or len(self.running_request_list) == 0:
            self._get_next_batch()
        # updata status after one inference step
        self._update_status_after_one_itreation()
        # if all samples in batch is invalid status, a static batch is over
        if self.count_of_invalid_sample == self.batch_size:
            self._get_next_batch()

    def schedule(self) -> Tuple[List[EntryMetaData], int]:
        if self.dyn_batch and len(self.dyn_batch) > 1:
            self._determine_batch_size()
        if self.batching_strategy == 'static':
            self._static_batch()
        elif not self.config.model_config.page_attention and self.batching_strategy == 'continuous':
            self._continuous_batch()
        elif self.config.model_config.fastserve:
            self._continuous_batch_fastserve()
        elif self.config.model_config.page_attention:  # 加入PA
            self._continuous_batch_pa()
        else:
            raise ValueError("Invalid batching strategy!, please setting static or continuous")
        return self.running_request_list, self.batch_size

    # 增加对 PA的_finished_request处理
    def _finished_pa_request(self, index, token, eos_id):
        # eos
        if token == eos_id:
            logging.debug("a request finished, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        # max len
        entry_data = self.running_request_list[index].get_entry_data()
        if entry_data.max_token_len <= entry_data.get_output_len():
            logging.debug("a request reached the max generate token length")
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        if entry_data.get_len() >= self.config.model_config.max_generate_length:
            logging.debug("a request reached seq len: %s, index: %s", self.config.max_generate_length, index)
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

        # input outofrange
        if entry_data.status == EntryStatus.INPUT_OUTOFRANGE or entry_data.status == EntryStatus.EMPTY_PROMPT_TOKEN:
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return
        # predict failed
        if token == -1:
            logging.debug("a request predict failed, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            self.running_request_list[index].cache_engine.release_cache()
            self.running_request_list[index].cache_engine.assign_null_block()
            return

    def _finished_request(self, index, token, eos_id):
        # eos
        if token == eos_id:
            logging.debug("a request finished, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

        # max len
        entry_data = self.running_request_list[index].get_entry_data()
        if entry_data.max_token_len <= entry_data.get_output_len():
            logging.debug("a request reached the max generate token length")
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_LENGTH_CAPPED)
            return

        # input outofrange
        if entry_data.status == EntryStatus.INPUT_OUTOFRANGE or entry_data.status == EntryStatus.EMPTY_PROMPT_TOKEN:
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return
        # predict failed
        if token == -1:
            logging.debug("a request predict failed, token equal to {}".format(token))
            self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return

    def upate_entries_after_one_step(self, outputs: List[int], eos_id: int, index_list: List[int] = None):
        """update status after ever iteration"""
        # optimize prefill multi-batch later
        if index_list is not None:
            # idx: index_list and outputs data index, index: batch list index.
            for idx, index in enumerate(index_list):
                self.running_request_list[index].is_prompt = False
                # invalid prompt
                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.PADDING_INVAILED:
                    continue

                if self.running_request_list[index].get_entry_data().get_status() == EntryStatus.INPUT_OUTOFRANGE:
                    update_token = INPUT_OUT_OF_TOKEN[0]
                elif self.running_request_list[index].get_entry_data().get_status() == EntryStatus.EMPTY_PROMPT_TOKEN:
                    update_token = INPUT_EMPTY_TOKEN[0]
                else:
                    update_token = outputs[idx]

                self.running_request_list[index].get_entry_data().updata_output_tokens(update_token)
                # valid prompt 区分PA处理
                if self.config.model_config.page_attention:
                    self._finished_pa_request(index, update_token, eos_id)
                else:
                    self._finished_request(index, update_token, eos_id)
        # decode
        else:
            for index, token in enumerate(outputs):
                if self.running_request_list[index].get_entry_data().get_status() != EntryStatus.RUNNING:  # 改动
                    continue
                # update new token to result list
                self.running_request_list[index].get_entry_data().updata_output_tokens(token)
                # 区分PA处理
                if self.config.model_config.page_attention:
                    self._finished_pa_request(index, token, eos_id)
                else:
                    self._finished_request(index, token, eos_id)

    def abort_entry(self,
                    request_id: str):
        for index, data in enumerate(self.running_request_list):
            if data.request_id == request_id:
                self.running_request_list[index].get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
