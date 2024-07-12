from typing import Any

from transformers import DataCollatorForLanguageModeling


class DebugDataCollator:
    def __init__(self, default_collator: DataCollatorForLanguageModeling):
        self.default_collator = default_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.default_collator(features)

        # Print or log the tokenized inputs here
        print("Input IDs:", batch["input_ids"])
        print("Attention Mask:", batch["attention_mask"])
        if "labels" in batch:
            print("Labels:", batch["labels"])

        return batch
