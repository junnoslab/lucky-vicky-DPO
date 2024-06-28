from peft import LoraConfig

from .model import ModelLoader, Models
from .train import Trainer


class Runner:
    def __init__(self):
        pass

    def run(self):
        loader = ModelLoader()

        config = LoraConfig()
        lora_model = loader.load_lora_model(
            Models.BLOSSOM, config=config, adapter_name="lora"
        )

        trainer = Trainer()
        trainer.train(
            model=lora_model, training_args=None, train_dataset=None, eval_dataset=None
        )
