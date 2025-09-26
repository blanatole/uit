## Phát hiện ảo giác tiếng Việt với Gemma-7B-IT (QLoRA) + Debate

Hệ thống phân loại ảo giác (hallucination) cho ngữ cảnh tiếng Việt với ba nhãn: `no`, `intrinsic`, `extrinsic`. Repo cung cấp:
- Tiền xử lý dữ liệu và tách tập theo tỉ lệ 80/10/10 (stratified)
- Fine-tune QLoRA cho `google/gemma-7b-it` tối ưu bộ nhớ với PyTorch 2.8+
- Đánh giá (Accuracy, Macro-F1) có thanh tiến trình và early stopping
- Suy luận theo hai cách:
  - Single-pass (baseline)
  - Debate 3 tác nhân (Literalist, Skeptic, Verifier) + Judge với prompts được cải thiện
- Resume training đầy đủ (optimizer, scheduler, learning rate states)
- Quản lý checkpoint tốt nhất tự động

### 1) Môi trường

Khuyến nghị GPU 48 GB (có thể thấp hơn nếu giảm batch/seq). Cài đặt:

```bash
pip install -r requirements.txt
# Tùy chọn: tối ưu cấp phát bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Tắt cảnh báo tokenizer parallelism
export TOKENIZERS_PARALLELISM=false
```

**Yêu cầu PyTorch ≥ 2.6** để có resume training đầy đủ. Thư viện chính: transformers, datasets, peft, bitsandbytes, scikit-learn, tqdm.

### 2) Dữ liệu

Các cột yêu cầu: `id, context, prompt, response, label`.
Tiền xử lý và tách tập (tạo `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`):

```bash
python scripts/preprocess_and_split.py --input data/vihallu-train.csv --outdir data
```

Lưu ý: `label ∈ {no|intrinsic|extrinsic}` (không phân biệt hoa/thường). Bộ tiền xử lý sẽ chuẩn hóa văn bản tiếng Việt.

### 3) Huấn luyện (QLoRA)

Huấn luyện được điều phối bằng script vòng lặp train → eval từng epoch, có early stopping và theo dõi checkpoint tốt nhất.

**Chế độ test nhanh (5 epochs, 100 train samples):**
```bash
scripts/train_eval_loop.sh --test
```

**Chế độ full (10 epochs, full dataset):**
```bash
scripts/train_eval_loop.sh
```

**Chế độ eval-only (chỉ đánh giá checkpoint tốt nhất):**
```bash
scripts/train_eval_loop.sh --eval-only
```

**Tính năng mới:**
- **Resume đầy đủ**: Epoch 1 từ scratch, các epoch sau resume từ checkpoint trước đó với đầy đủ optimizer, scheduler, learning rate states
- **Quản lý checkpoint tốt nhất**: Tự động lưu checkpoint có Macro-F1 cao nhất vào `out/gemma-vihallu/best`
- **Early stopping**: Dừng sau 2 epoch không cải thiện
- **Progress tracking**: Thanh tiến trình chi tiết cho cả training và evaluation

**Scripts chính:**
- `scripts/train_gemma_vihallu.py`: Huấn luyện 1 epoch (QLoRA 4-bit nf4, bf16, gradient checkpointing)
- `scripts/eval_gemma_vihallu.py`: Đánh giá với progress bar; ghi kết quả vào `results_dev.log`
- `scripts/train_eval_loop.sh`: Điều phối vòng lặp train/eval với early stopping

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
Dự đoán từ CSV bằng cơ chế debate với prompts được cải thiện (`rounds=1` nhanh; tăng để ổn định hơn):

```bash
python scripts/debate_to_csv.py \
  --ckpt out/gemma-vihallu/best \
  --input_csv data/vihallu-public-test.csv \
  --output_csv preds_vihallu_public_test_debate.csv \
  --rounds 1
```

**Cải tiến Debate:**
- **Prompts được cải thiện**: Rõ ràng hơn về vai trò của từng agent và quy tắc quyết định
- **Judge logic tốt hơn**: Quy tắc rõ ràng cho việc chọn nhãn dựa trên votes
- **Giảm bias**: Cân bằng hơn trong việc phân loại các nhãn
- **Test mode**: Có thể chạy với 20 mẫu để kiểm tra nhanh

Cả hai file CSV đều có: `id,predict_label` với `predict_label ∈ {no,intrinsic,extrinsic}`.

### 6) So sánh Single-pass vs Debate

**Đánh giá kết quả debate:**
```bash
python scripts/eval_debate_results.py \
  --pred_csv debate_test_full.csv \
  --test_jsonl data/test.jsonl
```

**So sánh với single-pass:**
```bash
# Single-pass evaluation
python scripts/eval_gemma_vihallu.py --ckpt out/gemma-vihallu/best --split test

# Debate evaluation  
python scripts/debate_to_csv.py --ckpt out/gemma-vihallu/best --input_csv data/test.jsonl --output_csv debate_test_full.csv --rounds 1
python scripts/eval_debate_results.py --pred_csv debate_test_full.csv --test_jsonl data/test.jsonl
```

**Scripts đánh giá:**
- `scripts/eval_debate_results.py`: Đánh giá kết quả debate với so sánh single-pass
- `scripts/predict_public_test.py`: Dự đoán trên public test set

### 7) Mẹo & Bộ nhớ

- Dùng `bf16=True`, lượng tử hóa 4-bit nf4, và gradient checkpointing (đã bật sẵn).
- Nếu OOM: giảm `per_device_train_batch_size`, tăng `gradient_accumulation_steps`, hoặc giảm `MAX_LEN`.
- Progress bar đã bật cho cả đánh giá và suy luận.
- Suy luận/đánh giá dùng giải mã quyết định (greedy), đã bỏ temperature.

### 8) Giới hạn đã biết

- **PyTorch < 2.6**: Resume đầy đủ optimizer/scheduler bị hạn chế. Vòng lặp vẫn resume trọng số; LR có thể reset mỗi epoch.
- **Debate performance**: Hiện tại debate có thể có bias về "intrinsic", cần cải thiện thêm prompts
- **Tốc độ**: Debate chậm hơn single-pass; tăng `rounds` sẽ chính xác hơn nhưng tốn thời gian
- **Memory**: Cần GPU 48GB cho full training; có thể giảm batch size cho GPU nhỏ hơn

### 9) Cấu trúc file chính

**Training & Evaluation:**
- `scripts/train_gemma_vihallu.py`: Huấn luyện 1 epoch với resume đầy đủ
- `scripts/train_eval_loop.sh`: Điều phối vòng lặp train/eval + early stopping + quản lý checkpoint tốt nhất
- `scripts/eval_gemma_vihallu.py`: Đánh giá single-pass với progress bar

**Inference:**
- `scripts/predict_public_test.py`: Dự đoán single-pass trên public test set
- `scripts/debate_to_csv.py`: Dự đoán debate với prompts cải thiện
- `scripts/eval_debate_results.py`: Đánh giá kết quả debate và so sánh với single-pass

**Data Processing:**
- `scripts/preprocess_and_split.py`: Làm sạch + tách train/val/test JSONL

### 10) Chạy nhanh để kiểm thử

```bash
# 1. Test mode nhanh (5 epochs, 100 samples)
scripts/train_eval_loop.sh --test

# 2. Đánh giá checkpoint tốt nhất
python scripts/eval_gemma_vihallu.py --ckpt out/gemma-vihallu/best --split val

# 3. Dự đoán public test (single-pass)
python scripts/predict_public_test.py \
  --input_csv data/vihallu-public-test.csv \
  --output_csv preds_vihallu_public_test_singlepass.csv \
  --model_dir out/gemma-vihallu/best

# 4. Dự đoán public test (debate) - test 20 mẫu trước
python scripts/debate_to_csv.py \
  --ckpt out/gemma-vihallu/best \
  --input_csv data/test.jsonl \
  --output_csv debate_test_20.csv \
  --rounds 1

# 5. Đánh giá kết quả debate
python scripts/eval_debate_results.py \
  --pred_csv debate_test_full.csv \
  --test_jsonl data/test.jsonl
```

### 11) Kết quả mẫu

**Training performance (test mode):**
- Accuracy: ~40-50%
- Macro-F1: ~35-45%
- Early stopping sau 2-3 epochs

**Debate vs Single-pass:**
- Debate có thể cải thiện accuracy nhưng cần tối ưu thêm prompts
- Single-pass nhanh hơn và ổn định hơn
- Cả hai đều có thể cải thiện với training tốt hơn



