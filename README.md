## Phát hiện ảo giác tiếng Việt với Gemma-7B-IT (QLoRA) + Debate

Hệ thống phân loại ảo giác (hallucination) cho ngữ cảnh tiếng Việt với ba nhãn: `no`, `intrinsic`, `extrinsic`. Repo cung cấp:
- Tiền xử lý dữ liệu và tách tập theo tỉ lệ 80/10/10 (stratified)
- Fine-tune QLoRA cho `google/gemma-7b-it` tối ưu bộ nhớ
- Đánh giá (Accuracy, Macro-F1) có thanh tiến trình
- Suy luận theo hai cách:
  - Single-pass (baseline)
  - Debate 3 tác nhân (Literalist, Skeptic, Verifier) + Judge

### 1) Môi trường

Khuyến nghị GPU 48 GB (có thể thấp hơn nếu giảm batch/seq). Cài đặt:

```bash
pip install -r requirements.txt
# Tùy chọn: tối ưu cấp phát bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Thư viện chính: transformers, datasets, peft, bitsandbytes, scikit-learn, tqdm.

### 2) Dữ liệu

Các cột yêu cầu: `id, context, prompt, response, label`.
Tiền xử lý và tách tập (tạo `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`):

```bash
python scripts/preprocess_and_split.py --input data/vihallu-train.csv --outdir data
```

Lưu ý: `label ∈ {no|intrinsic|extrinsic}` (không phân biệt hoa/thường). Bộ tiền xử lý sẽ chuẩn hóa văn bản tiếng Việt.

### 3) Huấn luyện (QLoRA)

Huấn luyện được điều phối bằng script vòng lặp train → eval từng epoch, có early stopping và theo dõi checkpoint tốt nhất.

Chế độ test nhanh:
```bash
scripts/train_eval_loop.sh --test
```
Chế độ full (early stop patience=2, tổng 10 epoch):
```bash
scripts/train_eval_loop.sh
```

Bên trong:
- `scripts/train_gemma_vihallu.py` huấn luyện 1 epoch mỗi lần gọi (QLoRA 4-bit nf4, bf16, gradient checkpointing).
- `scripts/eval_gemma_vihallu.py` đánh giá trên dev kèm progress bar; Accuracy/Macro-F1 ghi vào `results_dev.log`.
- Early stopping sau 2 epoch không cải thiện; in ra đường dẫn checkpoint tốt nhất.

Quan trọng:
- Do hạn chế bảo mật với torch cũ, việc resume chỉ chắc chắn khôi phục trọng số mô hình giữa các epoch. Trạng thái optimizer/scheduler có thể bị reset. Nếu cần resume đầy đủ, nâng PyTorch ≥ 2.6.

### 4) Đánh giá (single-pass)

Đánh giá checkpoint trên val/test (có progress bar):
```bash
python scripts/eval_gemma_vihallu.py --ckpt out/gemma-vihallu/checkpoint-<N> --split val
python scripts/eval_gemma_vihallu.py --ckpt out/gemma-vihallu/checkpoint-<N> --split test
```

In ra Accuracy, Macro-F1, báo cáo phân loại, và ma trận nhầm lẫn.

### 5) Suy luận

#### 5.1 Single-pass → CSV (2 cột: id, predict_label)
Dự đoán từ CSV gồm cột `id, context, prompt, response`:

```bash
python scripts/predict_to_csv.py \
  --ckpt out/gemma-vihallu/checkpoint-<BEST> \
  --input_csv data/vihallu-public-test.csv \
  --output_csv preds_vihallu_public_test_singlepass.csv
```

#### 5.2 Debate (3 tác nhân + Judge) → CSV
Dự đoán từ CSV bằng cơ chế debate (`rounds=1` nhanh; tăng để ổn định hơn):

```bash
python scripts/debate_to_csv.py \
  --ckpt out/gemma-vihallu/checkpoint-<BEST> \
  --input_csv data/vihallu-public-test.csv \
  --output_csv preds_vihallu_public_test_debate.csv \
  --rounds 1
```

Cả hai file CSV đều có: `id,predict_label` với `predict_label ∈ {no,intrinsic,extrinsic}`.

### 6) So sánh Single-pass vs Debate

Trên tập có nhãn (ví dụ `test.jsonl`):

```bash
python - <<'PY'
import sys, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
sys.path.append('scripts')
from debate_infer import debate_predict

CKPT = 'out/gemma-vihallu/checkpoint-<BEST>'
test_path = 'data/test.jsonl'

# single-pass
import subprocess
subprocess.run(['python','scripts/eval_gemma_vihallu.py','--ckpt',CKPT,'--split','test'])

# debate
preds = debate_predict(test_path, model_dir=CKPT, rounds=1)
df = pd.read_json(test_path, lines=True)
y_true = df['label'].str.lower().tolist()
labels = ['no','intrinsic','extrinsic']
print('Debate Accuracy:', accuracy_score(y_true, preds))
print('Debate Macro-F1:', f1_score(y_true, preds, average='macro', labels=labels))
print('Debate Report:\n', classification_report(y_true, preds, labels=labels, digits=4))
PY
```

### 7) Mẹo & Bộ nhớ

- Dùng `bf16=True`, lượng tử hóa 4-bit nf4, và gradient checkpointing (đã bật sẵn).
- Nếu OOM: giảm `per_device_train_batch_size`, tăng `gradient_accumulation_steps`, hoặc giảm `MAX_LEN`.
- Progress bar đã bật cho cả đánh giá và suy luận.
- Suy luận/đánh giá dùng giải mã quyết định (greedy), đã bỏ temperature.

### 8) Giới hạn đã biết

- Với PyTorch < 2.6, resume đầy đủ optimizer/scheduler bị hạn chế. Vòng lặp vẫn resume trọng số; LR có thể reset mỗi epoch.
- Debate chậm hơn single-pass; tăng `rounds` sẽ chính xác hơn nhưng tốn thời gian.

### 9) Cấu trúc file chính

- `scripts/preprocess_and_split.py`: Làm sạch + tách train/val/test JSONL
- `scripts/train_gemma_vihallu.py`: Huấn luyện 1 epoch (được gọi bởi vòng lặp)
- `scripts/train_eval_loop.sh`: Điều phối train/eval + early stopping
- `scripts/eval_gemma_vihallu.py`: Đánh giá (Accuracy, Macro-F1) có progress bar
- `scripts/predict_to_csv.py`: Dự đoán single-pass ra CSV
- `scripts/debate_infer.py`: Suy luận debate (dùng trong code)
- `scripts/debate_to_csv.py`: Dự đoán debate từ CSV → CSV

### 10) Chạy nhanh để kiểm thử

```bash
# vòng lặp nhanh (mẫu nhỏ) để xem progress bar
scripts/train_eval_loop.sh --test

# đánh giá checkpoint tốt nhất trên val
python scripts/eval_gemma_vihallu.py --ckpt out/gemma-vihallu/checkpoint-<BEST> --split val

# xuất dự đoán public test (single-pass)
python scripts/predict_to_csv.py --ckpt out/gemma-vihallu/checkpoint-<BEST> \
  --input_csv data/vihallu-public-test.csv --output_csv preds_vihallu_public_test_singlepass.csv

# xuất dự đoán public test (debate)
python scripts/debate_to_csv.py --ckpt out/gemma-vihallu/checkpoint-<BEST> \
  --input_csv data/vihallu-public-test.csv --output_csv preds_vihallu_public_test_debate.csv --rounds 1
```



