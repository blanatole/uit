#!/bin/bash

# Train-eval loop pipeline để tránh OOM và track best checkpoint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Detect if script is sourced; use return instead of exit to avoid killing shell
_IS_SOURCED=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _IS_SOURCED=1
fi
safe_exit() {
  local code=${1:-0}
  if [ "$_IS_SOURCED" -eq 1 ]; then
    return $code
  else
    exit $code
  fi
}

# Parse args: --test and optional --eval-only, --split={val|test}
MODE_TEST=0
MODE_EVAL_ONLY=0
EVAL_SPLIT="val"
for arg in "$@"; do
  case $arg in
    --test)
      MODE_TEST=1
      ;;
    --eval-only)
      MODE_EVAL_ONLY=1
      ;;
    --split=val)
      EVAL_SPLIT="val"
      ;;
    --split=test)
      EVAL_SPLIT="test"
      ;;
  esac
done

# Check for test mode
if [ $MODE_TEST -eq 1 ]; then
    TOTAL_EPOCHS=5
    RESULTS_FILE="results_dev_test.log"
    echo "🧪 TEST MODE: 5 epochs, 100 train samples, 50 val samples"
else
    TOTAL_EPOCHS=10
    RESULTS_FILE="results_dev.log"
    echo "🚀 PRODUCTION MODE: 10 epochs, full dataset"
fi

echo "📝 Results will be saved to $RESULTS_FILE"

# Clear previous results
> $RESULTS_FILE

# Variables for early stopping
BEST_MACRO_F1=0.0
PATIENCE_COUNT=0
PATIENCE=2
BEST_CHECKPOINT=""

# Eval-only mode: just evaluate latest checkpoint and exit
if [ $MODE_EVAL_ONLY -eq 1 ]; then
  # Try to read best checkpoint from results file first
  if [ -f "$RESULTS_FILE" ]; then
    # Robustly extract path after 'Best checkpoint: '
    BEST_FROM_LOG=$(awk -F"Best checkpoint: " '/Best checkpoint:/ {print $2}' "$RESULTS_FILE" | awk '{print $1}' | tail -n 1)
  else
    BEST_FROM_LOG=""
  fi
  # If not found, try the other results file
  if [ -z "$BEST_FROM_LOG" ]; then
    ALT_FILE=$([ $MODE_TEST -eq 1 ] && echo "results_dev.log" || echo "results_dev_test.log")
    if [ -f "$ALT_FILE" ]; then
      BEST_FROM_LOG=$(awk -F"Best checkpoint: " '/Best checkpoint:/ {print $2}' "$ALT_FILE" | awk '{print $1}' | tail -n 1)
    fi
  fi
  # Prefer dedicated 'best' checkpoint directory if exists
  if [ -d "/root/uit/out/gemma-vihallu/best" ]; then
    CKPT="/root/uit/out/gemma-vihallu/best"
  elif [ -n "$BEST_FROM_LOG" ]; then
    CKPT="$BEST_FROM_LOG"
  else
    CKPT=$(ls -dt /root/uit/out/gemma-vihallu/checkpoint-* 2>/dev/null | head -n 1)
  fi
  if [ -z "$CKPT" ]; then
    echo "❌ No checkpoint found for eval-only mode."
    safe_exit 1
  fi
  echo "🧭 Eval-only: using checkpoint $CKPT"
  if [ $MODE_TEST -eq 1 ]; then
      python /root/uit/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split $EVAL_SPLIT --test_mode 2>&1 | tee -a $RESULTS_FILE
  else
      python /root/uit/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split $EVAL_SPLIT 2>&1 | tee -a $RESULTS_FILE
  fi
  echo "✅ Eval-only completed."
  safe_exit 0
fi

if [ $MODE_EVAL_ONLY -eq 0 ]; then
for E in $(seq 1 $TOTAL_EPOCHS); do
    echo ""
    echo "=" "EPOCH $E/$TOTAL_EPOCHS" "="
    
    # Train thêm 1 epoch (Trainer sẽ tự resume từ checkpoint mới nhất)
    echo "🏋️ Training epoch $E..."
    if [ $MODE_TEST -eq 1 ]; then
        if [ $E -eq 1 ]; then
            python /root/uit/scripts/train_gemma_vihallu.py --test_mode --from_scratch
        else
            python /root/uit/scripts/train_gemma_vihallu.py --test_mode
        fi
    else
        if [ $E -eq 1 ]; then
            python /root/uit/scripts/train_gemma_vihallu.py --from_scratch
        else
            python /root/uit/scripts/train_gemma_vihallu.py
        fi
    fi
    # Tìm checkpoint mới nhất sau epoch vừa xong
    CKPT=$(ls -dt /root/uit/out/gemma-vihallu/checkpoint-* 2>/dev/null | head -n 1)
    
    if [ -z "$CKPT" ]; then
        echo "❌ No checkpoint found after epoch $E"
        continue
    fi
    
    echo "📁 Latest checkpoint: $CKPT"
    
    # Đánh giá trên dev (nhẹ VRAM vì chỉ generate)
    echo "🔍 Evaluating on validation set..."
    echo "=== EPOCH $E CHECKPOINT: $CKPT ===" >> $RESULTS_FILE
    
    # Capture eval output and extract Macro-F1
    # Chạy eval và hiển thị progress bar trực tiếp, đồng thời lưu vào file
    if [ "$1" = "--test" ]; then
        python /root/uit/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split val --test_mode 2>&1 | tee -a $RESULTS_FILE
    else
        python /root/uit/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split val 2>&1 | tee -a $RESULTS_FILE
    fi
    echo "" >> $RESULTS_FILE
    
    # Lấy output cuối cùng để extract Macro-F1
    EVAL_OUTPUT=$(tail -20 $RESULTS_FILE)
    
    # Extract Macro-F1 from output
    MACRO_F1=$(echo "$EVAL_OUTPUT" | grep "Macro-F1:" | awk '{print $2}')
    
    if [ -n "$MACRO_F1" ]; then
        echo "📊 Current Macro-F1: $MACRO_F1"
        echo "🏆 Best Macro-F1 so far: $BEST_MACRO_F1"
        
        # Check if current F1 is better than best (không phụ thuộc bc)
        if python -c "import sys; a=float('$MACRO_F1'); b=float('$BEST_MACRO_F1'); sys.exit(0 if a>b else 1)"; then
            BEST_MACRO_F1=$MACRO_F1
            BEST_CHECKPOINT=$CKPT
            PATIENCE_COUNT=0
            echo "🎉 New best Macro-F1: $BEST_MACRO_F1"
            echo "💾 Best checkpoint: $BEST_CHECKPOINT"
            # Sync to persistent 'best' directory
            BEST_DIR="/root/uit/out/gemma-vihallu/best"
            echo "📝 Updating best checkpoint copy at $BEST_DIR"
            mkdir -p "$BEST_DIR"
            rsync -a --delete "$BEST_CHECKPOINT/" "$BEST_DIR/"
        else
            PATIENCE_COUNT=$((PATIENCE_COUNT + 1))
            echo "⏳ No improvement. Patience: $PATIENCE_COUNT/$PATIENCE"
        fi
        
        # Check early stopping
        if [ $PATIENCE_COUNT -ge $PATIENCE ]; then
            echo "🛑 Early stopping triggered! No improvement for $PATIENCE epochs."
            echo "🏆 Best checkpoint: $BEST_CHECKPOINT (Macro-F1: $BEST_MACRO_F1)"
            break
        fi
    else
        echo "⚠️ Could not extract Macro-F1 from evaluation output"
    fi
    
    echo "✅ Epoch $E completed"
done
fi

echo ""
echo "🎉 Training completed! Check $RESULTS_FILE for all results"
echo "🏆 Best checkpoint: $BEST_CHECKPOINT (Macro-F1: $BEST_MACRO_F1)"
echo ""
BEST_FOR_TEST=$BEST_CHECKPOINT
if [ -d "/root/uit/out/gemma-vihallu/best" ]; then
  BEST_FOR_TEST="/root/uit/out/gemma-vihallu/best"
fi
echo "📊 To evaluate best model on test set:"
echo "   python /root/uit/scripts/eval_gemma_vihallu.py --ckpt \"$BEST_FOR_TEST\" --split test"
echo ""
echo "💡 Usage:"
echo "   Test mode:  . /root/uit/scripts/train_eval_loop.sh --test"
echo "   Full mode:  . /root/uit/scripts/train_eval_loop.sh"
echo "   Eval-only:  . /root/uit/scripts/train_eval_loop.sh --eval-only [--test]"
