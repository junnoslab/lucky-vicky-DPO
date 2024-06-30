import logging

from transformers import HfArgumentParser

from .runner import Runner
from .utils import TrainConfig

_LOGGER = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(TrainConfig)
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=args.logger_level)
    _LOGGER.log(
        level=999, msg=f"Logger level set to {logging.getLevelName(args.logger_level)}"
    )

    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
