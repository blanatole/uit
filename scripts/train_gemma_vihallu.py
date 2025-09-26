import json, re
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import torch
import os
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "google/gemma-7b-it"
# Đặt chiều dài hợp lý để cắt FLOPs padding thừa
MAX_LEN = 1536


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
        # Không pad ở đây; để collator pad động theo batch
        return tokenizer(
            examples["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
    
    ds_train = ds["train"].map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
    ds_val = ds["validation"].map(tokenize_fn, batched=True, remove_columns=ds["validation"].column_names)
    
    return ds_train, ds_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Test mode: 1 epoch, 50 train samples, 25 val samples")
    parser.add_argument("--from_scratch", action="store_true", help="Force training from scratch (ignore existing checkpoints)")
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    # Bật TF32 và chọn attention kernel nhanh
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass
    
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
        # Throughput cao hơn với effective batch tương tự
        per_device_train_batch_size = 6
        gradient_accumulation_steps = 4
        logging_steps = 10
        save_steps = 500

    # Kiểm tra checkpoint gần nhất để resume chính xác (bao gồm optimizer/scheduler/LR)
    output_dir = Path("out/gemma-vihallu")
    last_checkpoint = None

    def _latest_valid_checkpoint(dir_path: Path):
        # Lấy danh sách checkpoint-* và chọn cái có trainer_state.json tồn tại
        cks = sorted(dir_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for ck in cks:
            if (ck / "trainer_state.json").exists():
                return str(ck)
        return None

    if (not args.from_scratch) and output_dir.exists():
        # Thử API mặc định trước
        try:
            guessed = get_last_checkpoint(str(output_dir))
        except Exception:
            guessed = None
        # Xác thực checkpoint đoán được; nếu lỗi thì tìm checkpoint hợp lệ gần nhất
        if guessed and (Path(guessed) / "trainer_state.json").exists():
            last_checkpoint = guessed
        else:
            last_checkpoint = _latest_valid_checkpoint(output_dir)
            if guessed and last_checkpoint != guessed:
                print(f"⚠️ Detected invalid checkpoint {guessed}; falling back to {last_checkpoint}")
    
    # Xác định số epoch mục tiêu: nếu đã có checkpoint thì chạy thêm đúng 1 epoch
    target_num_epochs = 1
    if (not args.from_scratch) and last_checkpoint:
        try:
            import json, math
            state_path = Path(last_checkpoint) / "trainer_state.json"
            if state_path.exists():
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                prev_epoch = state.get("epoch")
                if isinstance(prev_epoch, (int, float)):
                    target_num_epochs = math.floor(prev_epoch) + 1
        except Exception as e:
            print(f"⚠️ Could not read previous epoch from trainer_state.json: {e}")

    train_args = TrainingArguments(
        output_dir="out/gemma-vihallu",
        num_train_epochs=target_num_epochs,
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
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        group_by_length=True,
        eval_accumulation_steps=1,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_cfg)
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"⚠️ Could not enable gradient checkpointing: {e}")

    # Collator pad động theo bội số 8 (tối ưu kernel); tự tạo labels từ input_ids
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

    # Trainer đơn giản chỉ để train (không eval)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        data_collator=collator,
    )
    
    # Training: resume chính xác nếu có checkpoint
    if (not args.from_scratch) and last_checkpoint:
        print(f"🔄 Resuming training from latest checkpoint: {last_checkpoint}")
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except KeyError as e:
            # Một số optimizer (vd 8-bit) có thể thiếu state khi resume; fallback resume một phần
            print(f"⚠️ Optimizer state missing ({e}). Resuming weights only...")
            trainer.train()
    else:
        print("🚀 Starting training from scratch...")
        trainer.train()

    # Hiển thị epoch đã hoàn thành để vòng lặp ngoài theo dõi
    try:
        tr_state = trainer.state
        print({
            'train_runtime': tr_state.train_runtime,
            'train_samples_per_second': tr_state.train_samples_per_second,
            'train_steps_per_second': tr_state.train_steps_per_second,
            'train_loss': float(getattr(tr_state, 'loss', 0.0)) if hasattr(tr_state, 'loss') else 0.0,
            'epoch': float(getattr(tr_state, 'epoch', target_num_epochs)),
        })
    except Exception:
        pass
    trainer.save_model("out/gemma-vihallu/adapter")
    tok.save_pretrained("out/gemma-vihallu/adapter")
    
    # Giải phóng GPU memory sau training
    torch.cuda.empty_cache()
    
    print("✅ Training completed!")