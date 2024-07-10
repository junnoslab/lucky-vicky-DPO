import logging
import os

from transformers import HfArgumentParser
import torch

from .runner import Runner
from .utils import TrainConfig

_LOGGER = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(TrainConfig)
    # HFParser with dataclass doesn't support choices, so we need to add it manually.
    parser.add_argument(
        "--train_mode", dest="train_mode", type=str, default="full", choices=["full", "sft", "dpo"]
    )
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=args.logger_level)
    _LOGGER.log(
        level=999, msg=f"Logger level set to {logging.getLevelName(args.logger_level)}"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.multiprocessing.set_start_method("spawn")

    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
