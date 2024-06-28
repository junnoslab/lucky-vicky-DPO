from datasets import Dataset

from .model import ModelLoader, Models
from .train import Trainer
from .utils import TrainConfig


class Runner:
    config: TrainConfig

    def __init__(self, config_args: TrainConfig):
        self.config = config_args

    def run(self):
        loader = ModelLoader()

        # 1. Load a LoraModel (Use LoraConfig)
        lora_model = loader.load_lora_model(Models.BLOSSOM, training_config=self.config)

        # 2. Load a dataset
        train_dataset = Dataset()
        eval_dataset = Dataset()

        # 3. Train (Use TrainingArguments)
        trainer = Trainer()
        trainer.train(
            model=lora_model, train_dataset=train_dataset, eval_dataset=eval_dataset
        )

        # 4. Save the model
