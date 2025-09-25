import json, re
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "google/gemma-7b-it"
MAX_LEN = 2048


def _format_messages(row: Dict) -> List[Dict[str, str]]:
    user = (
        "Bạn là bộ phân loại ảo giác. Nhiệm vụ: dựa trên NGỮ CẢNH và CÂU TRẢ LỜI của trợ lý, "
        "hãy phân loại duy nhất một nhãn trong {no, intrinsic, extrinsic}.\n\n"
        "Định nghĩa:\n"
        "- no: câu trả lời được hỗ trợ đầy đủ bởi ngữ cảnh.\n"
        "- intrinsic: mâu thuẫn/sai lệch so với ngữ cảnh (sai số liệu, thuộc tính, quan hệ...).\n"
        "- extrinsic: đưa thêm thông tin không có trong ngữ cảnh và không thể kiểm chứng từ ngữ cảnh.\n\n"
        f"NGỮ CẢNH:\n{row['context']}\n\n"
        f"CÂU HỎI:\n{row['prompt']}\n\n"
        f"CÂU TRẢ LỜI CỦA TRỢ LÝ:\n{row['response']}\n\n"
        "Chỉ in đúng một nhãn: no hoặc intrinsic hoặc extrinsic."
    )
    assistant = row["label"].strip().lower()
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def build_dataset(train_path: str, val_path: str, test_path: str, tokenizer: AutoTokenizer):
    ds = load_dataset("json", data_files={
        "train": train_path, "validation": val_path, "test": test_path
    })
    
    def to_text(x: Dict):
        messages = _format_messages(x)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    ds = ds.map(to_text)
    
    def tokenize_fn(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs
    
    ds_train = ds["train"].map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
    ds_val = ds["validation"].map(tokenize_fn, batched=True, remove_columns=ds["validation"].column_names)
    
    return ds_train, ds_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Test mode: 1 epoch, 50 train samples, 25 val samples")
    args = parser.parse_args()
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    # Test mode: chỉ dùng 50 mẫu train, 25 mẫu val để test nhanh hơn
    if args.test_mode:
        print("🧪 TEST MODE: 1 epoch, 50 train samples, 25 val samples")
        ds_train_full, ds_val_full = build_dataset("data/train.jsonl", "data/val.jsonl", "data/test.jsonl", tok)
        ds_train = ds_train_full.select(range(50))
        ds_val = ds_val_full.select(range(25))
    else:
        ds_train, ds_val = build_dataset("data/train.jsonl", "data/val.jsonl", "data/test.jsonl", tok)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_cfg, dtype=torch.bfloat16, device_map="auto"
    )
    
    # Ensure padding is defined for Trainer-internal tokenization if needed
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id
    
    # QLoRA: prepare for k-bit training (enable grads on inputs, cast norms, etc.)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    # Điều chỉnh config cho test mode
    if args.test_mode:
        per_device_train_batch_size = 2  # Nhỏ hơn cho test mode
        gradient_accumulation_steps = 2
        logging_steps = 5
        save_steps = 50
    else:
        per_device_train_batch_size = 4  # Tăng cho production
        gradient_accumulation_steps = 4
        logging_steps = 50
        save_steps = 500

    # Kiểm tra xem có checkpoint để resume không
    has_checkpoint = Path("out/gemma-vihallu").exists() and any(Path("out/gemma-vihallu").glob("checkpoint-*"))
    
    args = TrainingArguments(
        output_dir="out/gemma-vihallu",
        num_train_epochs=1,  # Chỉ train 1 epoch mỗi lần chạy
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,   # Nhỏ để tránh OOM
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="no",  # Không eval trong training loop
        save_strategy="epoch",
        save_total_limit=5,
        logging_steps=logging_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
        resume_from_checkpoint=has_checkpoint,  # Tự động resume nếu có checkpoint
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_cfg)
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"⚠️ Could not enable gradient checkpointing: {e}")

    # Collator với padding động để tránh lỗi độ dài tensor, đồng thời gán nhãn -100 cho padding
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Trainer đơn giản chỉ để train (không eval)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        data_collator=collator,
    )
    
    # Training sẽ tự động resume từ checkpoint nếu có
    if has_checkpoint:
        print("🔄 Resuming training from latest checkpoint...")
    else:
        print("🚀 Starting training from scratch...")
    
    trainer.train()
    trainer.save_model("out/gemma-vihallu/adapter")
    tok.save_pretrained("out/gemma-vihallu/adapter")
    
    # Giải phóng GPU memory sau training
    torch.cuda.empty_cache()
    
    print("✅ Training completed!")