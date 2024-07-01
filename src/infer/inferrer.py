from peft.peft_model import PeftModel
import torch.nn as nn


class Inferrer:
    def __init__(self):
        pass

    def infer(self, base_model: nn.Module):
        PeftModel.from_pretrained()
