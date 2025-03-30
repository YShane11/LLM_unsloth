from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import wandb
import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

wandb.login()
os.environ["WANDB_PROJECT"] = "YSH"

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    token = "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",
)
# 從 unsloth 下載 Llama-3.2-3B-Instruct 模型和相應的 tokenizer，設定最大序列長度和 4-bit 量化

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # 會影響模型壓縮的效果和表現
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16, # 調節低秩層對最終輸出的影響
    lora_dropout = 0.1,
    bias = "none", 
    use_gradient_checkpointing = True, 
    random_state = 3407,
    use_rslora = False, # 進一步降低記憶體的壓縮技術
    loftq_config = None, # 進一步減少模型的記憶體需求
)
# 使用 LoRA 進行模型壓縮與加速，配置梯度檢查點和目標模塊以提高記憶體效率

# ==============================================================處理dataset==================================================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        
        tokenized = tokenizer(text=text, truncation=False, add_special_tokens=True)
        if len(tokenized["input_ids"]) <= max_seq_length:
            texts.append(text)
        else:
            texts.append(None) 
    return { "text" : texts}

from datasets import load_dataset
dataset = load_dataset("YShane11/legislation_train", split = "train")
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda example: example["text"] is not None)
# ===========================================================================================================================================

torch.cuda.empty_cache()

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 50,
        num_train_epochs = 3, # Set this for 1 full training run.
        max_steps = 1000,
        learning_rate = 3e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()


model.save_pretrained("./YShane11/legislation") # Local saving
tokenizer.save_pretrained("./YShane11/legislation")
model.push_to_hub("YShane11/legislation", token = "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") # Online saving
tokenizer.push_to_hub("YShane11/legislation", token = "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") # Online saving


# if True: model.save_pretrained_gguf("/root/LLM_unsloth/gguf_models/YShane11", tokenizer, quantization_method = "f16")
# if True: model.push_to_hub_gguf("YShane11/llama3.2_flight", tokenizer, quantization_method = "f16", token = "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")
# if True: model.save_pretrained_gguf("/root/LLM_unsloth/gguf_models/YShane11", tokenizer, quantization_method = "q4_k_m")
# if True: model.push_to_hub_gguf("YShane11/llama3.2_flight", tokenizer, quantization_method = "q4_k_m", token = "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")
torch.cuda.empty_cache()

# 參見 https://github.com/unslothai/unsloth/wiki 獲取進階訓練技巧，如：
# (1) 儲存至GGUF / 合併至16位元以支援vLLM
# (2) 從已儲存的LoRA適配器繼續訓練
# (3) 添加評估迴圈 / 解決OOM問題
# (4) 自訂聊天範本
# python3 /root/LLM_unsloth/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile llama3.2_flight.gguf ./YShane11/llama3.2_flight