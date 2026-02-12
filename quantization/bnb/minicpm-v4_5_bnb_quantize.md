# BNB

:::{Note}
**Support:** MiniCPM-V4.5 / MiniCPM-V4.0 / MiniCPM-V2.6 / MiniCPM-V2.5
:::


## 1.Download the Model

Download the MiniCPM-V-4_5 model from [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4_5) and extract it to your local directory.

## 2.Quantization Script

The following script loads the original model, quantizes it to 4-bit using bitsandbytes, and saves the quantized model.

```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import GPUtil
import os

assert torch.cuda.is_available(), "CUDA is not available, but this code requires a GPU."

device = 'cuda'  # Select GPU
model_path = '/model/MiniCPM-V-4_5' # Model path
save_path = './model/MiniCPM-V-4_5-int4' # Path to save quantized model
image_path = './assets/airplane.jpeg'

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
    llm_int8_threshold=6.0
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    device_map=device,
    quantization_config=quantization_config,
    trust_remote_code=True
)

gpu_usage = GPUtil.getGPUs()[0].memoryUsed  
start = time.time()
response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "What is in this picture?"
        }
    ],
    tokenizer=tokenizer
)
print('Output after quantization:', response)
print('Inference time after quantization:', time.time() - start)
print(f"GPU memory usage after quantization: {round(gpu_usage/1024, 2)}GB")

# Save the quantized model and tokenizer
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
```

## 3.Expected Output

After quantization, you should see output similar to:

```
Output after quantization: The image depicts an Airbus A380-800 aircraft in mid-flight against a clear blue sky. The airplane is predominantly white with a distinctive blue tail fin featuring a red and white logo. The fuselage has the text "Airbus A380-800" written on it, and there are Chinese characters along the side of the aircraft. The landing gear is partially extended, indicating that the plane is either taking off or preparing to land. The engines are visible under the wings, and the overall design showcases the large size and advanced engineering of the Airbus A380 model.
Inference time after quantization: 9.316158771514893
GPU memory usage after quantization: 18.97GB
```

The quantized model will be saved in the directory specified by `save_path` and can be used for further fine-tuning or inference.

---
To customize the model path, image, or save path, modify the corresponding variables in the script.