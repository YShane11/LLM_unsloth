from unsloth import FastLanguageModel
# 引入 FastLanguageModel，用於快速載入和操作語言模型

from datasets import load_dataset
# 引入 load_dataset 函數，用於載入訓練資料集

from trl import SFTTrainer
# 引入 SFTTrainer，用於微調語言模型

from transformers import TrainingArguments, DataCollatorForSeq2Seq
# 引入 TrainingArguments（訓練參數設置）和 DataCollatorForSeq2Seq（數據整理工具）類別

from unsloth import is_bfloat16_supported
# 引入 is_bfloat16_supported 函數，用於檢查 bfloat16 格式的支援情況

import os
os.environ["WANDB_MODE"] = "disabled"

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",     
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  
    "unsloth/Mistral-Small-Instruct-2409",     
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",           

    "unsloth/Llama-3.2-1B-bnb-4bit",           
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]  # More models at https://huggingface.co/unsloth
# 定義可用的 4-bit 模型清單，這些模型使用低精度表示法以減少記憶體需求

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", 
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
    lora_alpha = 8, # 調節低秩層對最終輸出的影響
    lora_dropout = 0,
    bias = "none", 
    use_gradient_checkpointing = True, 
    random_state = 3407,
    use_rslora = False, # 進一步降低記憶體的壓縮技術
    loftq_config = None, # 進一步減少模型的記憶體需求
)
# 使用 LoRA 進行模型壓縮與加速，配置梯度檢查點和目標模塊以提高記憶體效率

# ==============================================================處理dataset==================================================================
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        # 應用聊天模板生成文本
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        
        # 確保 text 正確傳遞給 tokenizer
        tokenized = tokenizer(text=text, truncation=False, add_special_tokens=True)
        
        # 篩選 token 數量小於等於 1024 的文本
        if len(tokenized["input_ids"]) <= max_seq_length:
            texts.append(text)
        else:
            texts.append(None)  # 插入占位符，保證結構一致
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("YShane11/legislation", split = "train")


from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset = dataset.filter(lambda example: example["text"] is not None)
# ===========================================================================================================================================


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3, 
        max_steps = 100,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs",
        report_to = "wandb", 
    ),
)
# 建立 SFTTrainer 實例，設定模型和數據集，用於微調，並指定各項訓練參數

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)




trainer_stats = trainer.train()
# 開始訓練模型


# 參見 https://github.com/unslothai/unsloth/wiki 獲取進階訓練技巧，如：
# (1) 儲存至GGUF / 合併至16位元以支援vLLM
# (2) 從已儲存的LoRA適配器繼續訓練
# (3) 添加評估迴圈 / 解決OOM問題
# (4) 自訂聊天範本

model.push_to_hub("YShane11/llama3.2_legislation", token = "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") # Online saving
tokenizer.push_to_hub("YShane11/llama3.2_legislation", token = "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") # Online saving


if True: model.push_to_hub_gguf("YShane11/llama3.2_legislation", tokenizer, quantization_method = "f16", token = "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")
if True: model.push_to_hub_gguf("YShane11/llama3.2_legislation", tokenizer, quantization_method = "q4_k_m", token = "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")