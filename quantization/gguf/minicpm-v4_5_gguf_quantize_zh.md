# MiniCPM-V 4.5 - GGUF 量化指南

本指南将引导您完成将 MiniCPM-V 4.5 PyTorch 模型转换为 GGUF 格式并进行量化的过程。

本指南涉及的模型需要基于 [`llama.cpp`](../../deployment/llama.cpp/minicpm-v4_5_llamacpp_zh.md)/[`ollama`](../../deployment/ollama/minicpm-v4_5_ollama_zh.md) 使用。

### 1. 下载 PyTorch 模型

首先，请从以下任一来源获取原始的 PyTorch 模型文件：

*   **HuggingFace:** https://huggingface.co/openbmb/MiniCPM-V-4_5
*   **魔搭社区:** https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5

### 2. 将 PyTorch 模型转换为 GGUF 格式

请依次执行以下命令，以完成模型结构调整、视觉编码器转换和语言模型转换。

```bash
# 步骤 1: 对模型结构进行预处理
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4_5

# 步骤 2: 将视觉编码器转换为 GGUF 格式
python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4_5 --minicpmv-projector ../MiniCPM-V-4_5/minicpmv.projector --output-dir ../MiniCPM-V-4_5/ --minicpmv_version 6

# 步骤 3: 将语言模型转换为 GGUF 格式
python ./convert-hf-to-gguf.py ../MiniCPM-V-4_5/model
```

### 3. 执行 INT4 量化

转换完成后，使用 `llama-quantize` 工具对 F16 精度的 GGUF 模型进行 INT4 量化。

```bash
./llama-quantize ../MiniCPM-V-4_5/model/ggml-model-f16.gguf ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```
