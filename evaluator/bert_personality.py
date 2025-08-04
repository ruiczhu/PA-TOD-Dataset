from transformers import BertForSequenceClassification, BertTokenizer
import torch


class PersonalityDetector():
    def __init__(self, model_path: str = "../Model/Bert_personality", num_labels: int = 5, do_lower_case: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        self._configure_labels()

    def _configure_labels(self):
        self.model.config.label2id = {
            "Extroversion": 0,
            "Neuroticism": 1,
            "Agreeableness": 2,
            "Conscientiousness": 3,
            "Openness": 4,
        }
        self.model.config.id2label = {
            "0": "Extroversion",
            "1": "Neuroticism",
            "2": "Agreeableness",
            "3": "Conscientiousness",
            "4": "Openness",
        }

    def personality_detection(self, model_input: str) -> dict:
        if len(model_input) == 0:
            return {
                "Extroversion": float(0),
                "Neuroticism": float(0),
                " Agreeableness": float(0),
                "Conscientiousness": float(0),
                "Openness": float(0),
            }

        # 使用 tokenizer 编码输入，限制最大长度为 512
        inputs = self.tokenizer.encode_plus(
            model_input,
            max_length=512,  # BERT 最大支持长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断超长序列
            return_tensors='pt'  # 直接返回 PyTorch 张量
        )

        # 移动到模型所在的设备（CPU 或 GPU）
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids', None)  # 自动处理 token_type_ids
            )

        b_logit_pred = outs.logits  # 获取模型输出的 logits
        pred_label = torch.sigmoid(b_logit_pred)  # 应用 sigmoid 获取概率

        return {
            "Extroversion": float(pred_label[0][0]),  # 外向性概率
            "Neuroticism": float(pred_label[0][1]),  # 神经质概率
            "Agreeableness": float(pred_label[0][2]),  # 亲和性概率
            "Conscientiousness": float(pred_label[0][3]),  # 尽责性概率
            "Openness": float(pred_label[0][4]),  # 开放性概率
        }

