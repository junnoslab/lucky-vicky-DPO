# LoRA

Repository for PEFT Fine-Tuning with LoRA / QLoRA.

## Resources

### Models

- [MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
- ✅ [yanolja/EEVE-Korean-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0)

### Datasets

- [Junnos/luckyvicky](https://huggingface.co/datasets/Junnos/luckyvicky)

## How to start

I used [pixi](https://prefix.dev/) for dependency & project manager.
You can install pixi simply with command below

```shell
curl -fsSL https://pixi.sh/install.sh | bash
```

Once pixi is installed, use `pixi shell` to enable venv from project's directory.

```shell
pixi shell
```

Or, you can simply run training or inference by using pixi's task commands

```shell
## Training
# t is short-name for train.
# wandb setup is required.
pixi r t
pixi r train

# td is short-name for train-debug.
# Use this command to disable wandb.
pixi r td
pixi r train-debug

## Inference
# i is short-name for infer.
pixi r i
pixi r infer
```

## Result
