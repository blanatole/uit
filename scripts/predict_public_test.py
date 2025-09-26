#!/usr/bin/env python3
"""
Script để dự đoán trên vihallu-public-test.csv bằng best model
"""
import re, torch, json, argparse, sys
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

LABELS = ["no", "intrinsic", "extrinsic"]
PAT = re.compile(r"\b(no|intrinsic|extrinsic)\b", re.IGNORECASE)

def build_prompt(row):
    """Xây dựng prompt cho inference"""
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
def predict_public_test(input_csv, output_csv, model_dir):
    """Dự đoán trên public test set"""
    print(f"🔄 Loading model from {model_dir}...")
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    
    print(f"📂 Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"🔍 Predicting on {len(df)} samples...")
    print("=" * 50)
    
    predictions = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="🔍 Predicting", dynamic_ncols=True)
    
    for i, (_, row) in enumerate(pbar):
        msgs = [{"role": "user", "content": build_prompt(row)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        
        out = model.generate(
            **inputs, 
            max_new_tokens=4, 
            do_sample=False, 
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )
        
        dec = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = PAT.search(dec)
        y = m.group(1).lower() if m else "no"
        predictions.append(y)
        
        # Update progress bar
        pbar.set_postfix({'Pred': y, 'Sample': f'{i+1}/{len(df)}'})
    
    # Tạo output DataFrame
    df_output = pd.DataFrame({
        'id': df['id'],
        'predict_label': predictions
    })
    
    # Lưu kết quả
    df_output.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to {output_csv}")
    
    # Thống kê
    print("\n📊 Prediction distribution:")
    pred_counts = pd.Series(predictions).value_counts()
    for label in LABELS:
        count = pred_counts.get(label, 0)
        pct = count / len(predictions) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on vihallu-public-test.csv")
    parser.add_argument("--input_csv", type=str, default="data/vihallu-public-test.csv", help="Input CSV file")
    parser.add_argument("--output_csv", type=str, default="preds_vihallu_public_test.csv", help="Output CSV file")
    parser.add_argument("--model_dir", type=str, default="out/gemma-vihallu/best", help="Model directory")
    
    args = parser.parse_args()
    
    print(f"🔍 Predicting on: {args.input_csv}")
    print(f"📁 Using model: {args.model_dir}")
    print(f"💾 Output to: {args.output_csv}")
    
    predict_public_test(args.input_csv, args.output_csv, args.model_dir)
