import torch
from transformers import BertTokenizer
from models.networks import HZTypoCheckerNet

from typing import List
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CheckerOutput(OrderedDict):
    raw_tokens: List[str]
    check_cls: List[int]
    mod_tokens: List[str]


class HZTypoChecker(object):
    def __init__(self, model_path: str, tokenizer_path: str):

        self.model = self.load_model(model_path)
        self.model.eval()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    

    def load_model(self, model_path):
        mlm_name = f"{model_path}/mlm"
        cls_name = f"{model_path}/detector.pth"
        model = HZTypoCheckerNet(mlm_name)
        model.detector.load_state_dict(torch.load(cls_name))

        return model
    

    def check(self, txt: str) -> CheckerOutput:
        token_output = self.tokenizer(txt, return_tensors="pt")
        input_ids = token_output.input_ids
        attention_mask = token_output.attention_mask
        
        mlm_pred, cls_pred = self.model(input_ids, attention_mask)
        
        mlm_pred_ids = torch.argmax(mlm_pred, dim=-1)
        cls_pred = torch.gt(cls_pred.sigmoid(), 0.5).long()

        raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        mod_tokens = self.tokenizer.convert_ids_to_tokens(mlm_pred_ids[0])
        cls_pred = cls_pred[0]
        
        return CheckerOutput(
            raw_tokens=raw_tokens,
            check_cls=cls_pred,
            mod_tokens=mod_tokens
        )






