# LoRA

Repository for PEFT Fine-Tuning with LoRA / QLoRA.

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
# t is short for train. They're basically same.
pixi r t
pixi r train

# i is shortname for infer. They're same also.
pixi r i
pixi r infer
```
