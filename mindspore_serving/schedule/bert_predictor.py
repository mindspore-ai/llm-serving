from mindformers import AutoTokenizer, AutoConfig
from mindformers.models.bert import BertModel
import mindspore as ms
from mindspore import nn, load_checkpoint, load_param_into_net, Tensor, ops

class BertClassificationModel(nn.Cell):
    def __init__(self, config, hidden_dim, num_classes, fix_weight=True):
        super().__init__()
        self.config = config
        self.model = BertModel(config)
        # Fix the weights of the pretrained model
        if fix_weight:
            for param in self.model.get_parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Dense(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(hidden_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(axis=-1)

    def construct(self, input_ids, token_type_ids, input_mask):
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, input_mask=input_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs[0][:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.logsoftmax(self.fc2(output))
        return output
    
class Predictor:
    def __init__(self, hidden_dim, num_classes, model_name, ckpt_path=None, device_target="CPU", device_id=0):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device_target = device_target
        self.device_id = device_id
        self.model = self.init_model(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def init_model(self, ckpt_path):
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=self.device_target, device_id=self.device_id)
        model = BertClassificationModel(self.config, hidden_dim=self.hidden_dim, num_classes=self.num_classes)
        if ckpt_path:
            params = load_checkpoint(ckpt_file_name=ckpt_path)
            param_not_load, _ = load_param_into_net(net=model, parameter_dict=params)
            print("Loaded model weights from disk.")
            print(f"no laoded params:{param_not_load}")
        model.set_train(False)
        return model

    def tokenize_for_serving(self, question, falg_head_tail=False):
        example = self.tokenizer(question, truncation=True, padding="max_length", max_length=self.config.seq_length)
        example["input_mask"] =example["attention_mask"]
        if len(example['input_ids']) >= 512:
            if falg_head_tail:
                example['input_ids'] = example['input_ids'][: 128] + example['input_ids'][-384: ]
                example['token_type_ids'] = example['token_type_ids'][: 128] + example['token_type_ids'][-384: ]
                example['input_mask'] = example['attention_mask'][: 128] + example['attention_mask'][-384: ]
            else:
                example['input_ids'] = example['input_ids'][-512: ]
                example['token_type_ids'] = example['token_type_ids'][-512: ]
                example['input_mask'] = example['attention_mask'][-512: ]
        return example


    def predict(self, question):
        batch = self.tokenize_for_serving(question)
        input_ids = Tensor(batch['input_ids'], ms.int32).unsqueeze(0)
        input_mask = Tensor(batch['input_mask'], ms.int32).unsqueeze(0)
        token_type_ids = Tensor(batch['token_type_ids'], ms.int32).unsqueeze(0)
        predictions = self.model(input_ids=input_ids, token_type_ids=token_type_ids, input_mask=input_mask)
        predictions = ops.argmax(predictions, dim=-1)
        return predictions[0].item()