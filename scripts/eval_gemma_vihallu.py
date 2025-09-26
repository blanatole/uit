import re, torch, json, argparse, sys
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

LABELS = ["no", "intrinsic", "extrinsic"]
PAT = re.compile(r"\b(no|intrinsic|extrinsic)\b", re.IGNORECASE)


def build_prompt(row):
    return (
        "Bạn là bộ phân loại ảo giác. Nhiệm vụ: dựa trên NGỮ CẢNH và CÂU TRẢ LỜI của trợ lý, "
        "hãy phân loại duy nhất một nhãn trong {no, intrinsic, extrinsic}.\n\n"
        "Định nghĩa:\n"
        "- no: câu trả lời được hỗ trợ đầy đủ bởi ngữ cảnh.\n"
        "- intrinsic: mâu thuẫn/sai lệch so với ngữ cảnh.\n"
        "- extrinsic: đưa thêm thông tin không có trong ngữ cảnh và không thể kiểm chứng.\n\n"
        f"NGỮ CẢNH:\n{row['context']}\n\n"
        f"CÂU HỎI:\n{row['prompt']}\n\n"
        f"CÂU TRẢ LỜI CỦA TRỢ LÝ:\n{row['response']}\n\n"
        "Chỉ in đúng một nhãn: no hoặc intrinsic hoặc extrinsic."
    )


@torch.inference_mode()
def predict(df_path: str, model_dir: str, test_mode: bool = False):
    from tqdm import tqdm
    import time
    
    print(f"🔄 Loading model from {model_dir}...")
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    
    print(f"📂 Loading dataset from {df_path}...")
    df = pd.read_json(df_path, lines=True)
    
    # Test mode: chỉ dùng 25 samples
    if test_mode:
        df = df.head(25)
        print(f"🧪 TEST MODE: Using only {len(df)} samples")

    preds, gts = [], []
    print(f"🔍 Evaluating {len(df)} samples...")
    print("=" * 50)
    
    # Tạo progress bar với thông tin chi tiết hơn - force hiển thị
    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc="🔍 Evaluating",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        disable=False,
        dynamic_ncols=True,
        ascii=True,
    )
    
    for i, (_, row) in enumerate(pbar):
        start_time = time.time()
        
        msgs = [{"role": "user", "content": build_prompt(row)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        
        out = model.generate(
            **inputs, max_new_tokens=4, do_sample=False, eos_token_id=tok.eos_token_id
        )
        
        dec = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = PAT.search(dec)
        y = m.group(1).lower() if m else "no"
        preds.append(y)
        gts.append(row["label"].strip().lower())
        
        # Cập nhật description với thông tin thời gian thực
        elapsed = time.time() - start_time
        pbar.set_postfix({
            'Sample': f'{i+1}/{len(df)}',
            'Time': f'{elapsed:.2f}s',
            'Pred': y,
            'True': row["label"].strip().lower()
        })
        pbar.refresh()

    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro", labels=LABELS, zero_division=0)
    print("Accuracy:", acc)
    print("Macro-F1:", f1m)
    print("Report:\n", classification_report(gts, preds, labels=LABELS, digits=4, zero_division=0))
    print("Confusion:\n", confusion_matrix(gts, preds, labels=LABELS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gemma model on validation/test set")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory (e.g., out/gemma-vihallu/checkpoint-1000)")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--test_mode", action="store_true", help="Test mode: only evaluate on 25 samples")
    args = parser.parse_args()
    
    # Map split to file path
    split_files = {"val": "data/val.jsonl", "test": "data/test.jsonl"}
    df_path = split_files[args.split]
    
    print(f"🔍 Evaluating checkpoint: {args.ckpt}")
    print(f"📊 Dataset: {args.split} ({df_path})")
    if args.test_mode:
        print("🧪 TEST MODE: Only evaluating on 25 samples")
    
    predict(df_path, args.ckpt, test_mode=args.test_mode)