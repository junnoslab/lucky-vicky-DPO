from .data import DataLoader, Datasets
from .model import ModelLoader, Models
from .train import Trainer
from .utils import TrainConfig


class Runner:
    config: TrainConfig

    def __init__(self, config_args: TrainConfig):
        self.config = config_args

    def run(self):
        model_loader = ModelLoader()

        # 1. Load a LoraModel (Use LoraConfig)
        tokenizer, model = model_loader.load_tokenizer_and_model(Models.BLOSSOM)

        lora_model = model_loader.load_lora_model(
            Models.BLOSSOM, training_config=self.config
        )

        # 2. Load a dataset
        data_loader = DataLoader()

        dataset = data_loader.load_dataset(Datasets.LUCKY_VICKY)

        # 3. Train (Use TrainingArguments)
        trainer = Trainer(config=self.config)
        trainer.train(
            model=lora_model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
        )

        # 4. Save the model
