from unsloth import FastLanguageModel
import torch
import os
import subprocess
from transformers import AutoModelForCausalLM




# Hugging Face 模型名稱
hf_model_name = "YShane11/llama3.2_flight"
local_model_path = "/root/LLM_unsloth/YShane11/llama3.2_flight"  # 本地模型下載目錄
hf_token = "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"  # 請確保這是你的 Hugging Face Token

os.environ["HF_TOKEN"] = hf_token

# 下載完整 Hugging Face 模型
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name, 
    cache_dir=local_model_path, 
    force_download=False,
    token=hf_token
)

# 設定量化方法
quantization_methods = {
    "f16": "f16",
    "q4_k_m": "q8_0"
}

# 確保 gguf_models/ 目錄存在
output_dir = "/root/LLM_unsloth/gguf_models"
os.makedirs(output_dir, exist_ok=True)

# 確保 gguf_models/ 目錄有寫入權限
os.chmod(output_dir, 0o777)

# 釋放 CUDA 記憶體，避免 OOM 問題
torch.cuda.empty_cache()

# **手動轉換 GGUF**
for method, outtype in quantization_methods.items():
    print(f"🔄 正在量化 {hf_model_name} 為 {method} ...")

    output_file = os.path.join(output_dir, f"{hf_model_name}_{method}.gguf")

    # **確保模型目錄存在**
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"❌ 模型路徑不存在: {local_model_path}")

    # **使用 convert_hf_to_gguf_update.py 進行轉換**
    subprocess.run([
        "python3", "./llama.cpp/convert_hf_to_gguf_update.py",
        hf_token,
        local_model_path,  # 本地模型目錄
        "--outfile", output_file,
        "--outtype", outtype,
    ], check=True)

    print(f"✅ {method} 量化完成並儲存於 {output_file}！")

# 釋放記憶體
torch.cuda.empty_cache()
print("🎉 所有 GGUF 量化完成！")
