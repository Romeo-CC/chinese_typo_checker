import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.activations import get_activation
from typing import Tuple, Optional


class HZTypoDetectionHead(nn.Module):
    """Predict whether a token is the right one in the placexs"""
    def __init__(self, config):
        super().__init__()
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activaton = get_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activaton(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.cls(hidden_states).squeeze(-1)

        return logits



class HZTypoCheckerNet(nn.Module):
    """Construct Chinese Hanzi typo checker based on BERT Model"""
    def __init__(self, model_name):
        super().__init__()

        self.corrector = BertForMaskedLM.from_pretrained(model_name)
        self.config = self.corrector.config
        self.detector = HZTypoDetectionHead(self.config)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        outputs = self.corrector(
            input_ids, attention_mask, 
            output_hidden_states = True
        )

        hidden_states = outputs.hidden_states

        mlm_pred = outputs.logits

        last_hidden_states = hidden_states[-1]

        det_perd = self.detector(last_hidden_states)

        return mlm_pred, det_perd


        
