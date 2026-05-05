# StableI2I
Official implementation of **StableI2I: Spotting Unintended Changes in Image-to-Image Transition** (ICML 2026)

Any questions can be consulted -> (Email:lijiayang.cs@gmail.com)

Looking forward to your ⭐！

### 📌 TODOs
> - [X] release code  
> - [X] release ckpt
> - [ ] release pip-pkg
> - [ ] release arxiv
> - [ ] ICML version paper

## Core Concept:
StableI2I is a vision-language evaluator for image-to-image generation. Given a before image, an after image, and an editing prompt, it checks whether the intended edit is correct while preserved regions remain stable.

The current release focuses on three aspects:
- `Semantic`: unintended additions, removals, or replacements in preserved regions
- `Structure`: misalignment or repainting in preserved regions
- `Low-Level`: blur, noise, color cast, exposure degradation, or artifacts in preserved regions

## Environment Setting:
Install dependencies:

```bash
pip install -r requirements.txt
```

The runtime is based on Qwen3-VL, so the environment should follow the official Qwen3-VL setup for CUDA, model weights, and Transformers compatibility.

Useful environment variables:

```bash
set MODEL_PATH=path\to\ckpt
set GPU_ID=0
```

## APP Usage:
`app.py` is the local web demo and API entry. Running it starts a FastAPI service with a browser UI.

Example:

```bash
set MODEL_PATH=path\to\ckpt
set GPU_ID=0
set HOST=127.0.0.1
set PORT=10004
python app.py
```

Then open:

```text
http://127.0.0.1:10004
```

The demo supports:
- built-in examples
- inference by local image path
- inference by image upload
- summarized semantic / structure / low-level results

## inference
See [infer.md](./infer.md).
## Training
Training code is not separately packaged in this repository yet.

Recommended official references:
- Qwen3-VL: [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- Swift: [modelscope/ms-swift](https://github.com/modelscope/ms-swift)

Notes:
- For SFT, start from the official Qwen3-VL finetuning workflow.
- For GRPO and related alignment training, use Swift.
