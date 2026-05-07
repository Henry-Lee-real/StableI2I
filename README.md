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

[![HuggingFace](https://img.shields.io/badge/HuggingFace-StableI2I-ffcc4d?logo=huggingface&logoColor=white&style=flat)](https://huggingface.co/collections/lijiayangCS/stablei2i)
[![Project Page](https://img.shields.io/badge/Project%20Page-StableI2I-blue?style=flat)](https://henry-lee-real.github.io/StableI2I_Page/)
[![arXiv 2605.04453](https://img.shields.io/badge/arXiv-2605.04453-b31b1b?logo=arXiv&logoColor=white&style=flat)](https://arxiv.org/pdf/2605.04453)

## Core Concept:
In most real-world image-to-image (I2I) scenarios, existing evaluations primarily focus on instruction following and the perceptual quality or aesthetics of the generated images. However, they largely fail to assess whether the output image preserves the semantic correspondence and spatial structure of the input image. To address this limitation, we propose StableI2I, a unified and dynamic evaluation framework that explicitly measures content fidelity and pre--post consistency across a wide range of I2I tasks without requiring reference images, including image editing and image restoration. In addition, we construct StableI2I-Bench, a benchmark designed to systematically evaluate the accuracy of MLLMs on such fidelity and consistency assessment tasks. Extensive experimental results demonstrate that StableI2I provides accurate, fine-grained, and interpretable evaluations of content fidelity and consistency, with strong correlations to human subjective judgments. Our framework serves as a practical and reliable evaluation tool for diagnosing content consistency and benchmarking model performance in real-world I2I systems.

## Environment Setting:
Install dependencies:

```bash
pip install -r requirements.txt
```

The specific environment is consistent with that of Qwen3-VL.


## APP Usage:
<img width="1974" height="1211" alt="image" src="https://github.com/user-attachments/assets/8104c802-58c4-4b63-bb44-f86158f960d2" />


`app.py` is the local web demo and API entry. Running it starts a FastAPI service with a browser UI.

Example:

```bash
set MODEL_PATH=path/to/ckpt
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

## Inference
See [infer.md](./infer.md).

## Training

Recommended official references:
- Qwen3-VL: [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- Swift: [modelscope/ms-swift](https://github.com/modelscope/ms-swift)

Notes:
- For SFT, start from the official Qwen3-VL finetuning workflow.
- For GRPO and related alignment training, use Swift.
