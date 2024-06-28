from transformers import HfArgumentParser

from .runner import Runner
from .utils import TrainConfig


def main():
    parser = HfArgumentParser(TrainConfig)
    args = parser.parse_args_into_dataclasses()[0]

    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
