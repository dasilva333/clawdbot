# MiniCPM-V 4.5 - llama.cpp

## 1. Build llama.cpp

Clone the llama.cpp repository:
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Build llama.cpp using `CMake`: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

**CPU/Metal:**
```bash
cmake -B build
cmake --build build --config Release
```

**CUDA:**
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```
## 2. GGUF files

### Option 1: Download official GGUF files

Download converted language model file (e.g., `ggml-model-Q4_K_M.gguf`) and vision model file (`mmproj-model-f16.gguf`) from:
*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf
*   ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-gguf

### Option 2: Convert from PyTorch model

Download the MiniCPM-V-4_5 PyTorch model to "MiniCPM-V-4_5" folder:
*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4_5
*   ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5

Convert the PyTorch model to GGUF:

```bash
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4_5

python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4_5 --minicpmv-projector ../MiniCPM-V-4_5/minicpmv.projector --output-dir ../MiniCPM-V-4_5/ --minicpmv_version 6

python ./convert_hf_to_gguf.py ../MiniCPM-V-4_5/model

# quantize int4 version
./llama-quantize ../MiniCPM-V-4_5/model/Model-3.6B-F16.gguf ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

## 3. Model Inference

```bash
cd build/bin/

# run f16 version
./llama-mtmd-cli -m ../MiniCPM-V-4_5/model/Model-3.6B-F16.gguf --mmproj ../MiniCPM-V-4_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run quantized int4 version
./llama-mtmd-cli -m ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# or run in interactive mode
./llama-mtmd-cli -m ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i

# run with reasoning enabled (think mode without token limit)
./llama-mtmd-cli -m ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg --jinja --reasoning-budget -1 -p "what is it?"

# run with reasoning disabled (no think mode)
./llama-mtmd-cli -m ../MiniCPM-V-4_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg --jinja --reasoning-budget 0 -p "what is it?"


```

**Argument Reference:**

| Argument | `-m, --model` | `--mmproj` | `--image` | `-p, --prompt` | `-c, --ctx-size` | `--reasoning-budget` | `--jinja` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Description | Path to the language model | Path to the vision model | Path to the input image | The prompt | Maximum context size | Maximum tokens for model reasoning (-1 for unlimited, 0 for disabled) | Enable Jinja template rendering |
