#!/bin/bash

# Train-eval loop pipeline để tránh OOM và track best checkpoint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check for test mode
if [ "$1" = "--test" ]; then
    TOTAL_EPOCHS=2
    RESULTS_FILE="results_dev_test.log"
    echo "🧪 TEST MODE: 2 epochs, 100 train samples, 50 val samples"
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

for E in $(seq 1 $TOTAL_EPOCHS); do
    echo ""
    echo "=" "EPOCH $E/$TOTAL_EPOCHS" "="
    
    # Train thêm 1 epoch (resume tự động)
    echo "🏋️ Training epoch $E..."
    if [ "$1" = "--test" ]; then
        python /root/uit-dsc/scripts/train_gemma_vihallu.py --test_mode
    else
        python /root/uit-dsc/scripts/train_gemma_vihallu.py
    fi
    
    # Tìm checkpoint mới nhất sau epoch vừa xong
    CKPT=$(ls -dt /root/uit-dsc/out/gemma-vihallu/checkpoint-* 2>/dev/null | head -n 1)
    
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
        python /root/uit-dsc/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split val --test_mode 2>&1 | tee -a $RESULTS_FILE
    else
        python /root/uit-dsc/scripts/eval_gemma_vihallu.py --ckpt "$CKPT" --split val 2>&1 | tee -a $RESULTS_FILE
    fi
    echo "" >> $RESULTS_FILE
    
    # Lấy output cuối cùng để extract Macro-F1
    EVAL_OUTPUT=$(tail -20 $RESULTS_FILE)
    
    # Extract Macro-F1 from output
    MACRO_F1=$(echo "$EVAL_OUTPUT" | grep "Macro-F1:" | awk '{print $2}')
    
    if [ -n "$MACRO_F1" ]; then
        echo "📊 Current Macro-F1: $MACRO_F1"
        echo "🏆 Best Macro-F1 so far: $BEST_MACRO_F1"
        
        # Check if current F1 is better than best
        if (( $(echo "$MACRO_F1 > $BEST_MACRO_F1" | bc -l) )); then
            BEST_MACRO_F1=$MACRO_F1
            BEST_CHECKPOINT=$CKPT
            PATIENCE_COUNT=0
            echo "🎉 New best Macro-F1: $BEST_MACRO_F1"
            echo "💾 Best checkpoint: $BEST_CHECKPOINT"
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

echo ""
echo "🎉 Training completed! Check $RESULTS_FILE for all results"
echo "🏆 Best checkpoint: $BEST_CHECKPOINT (Macro-F1: $BEST_MACRO_F1)"
echo ""
echo "📊 To evaluate best model on test set:"
echo "   python /root/uit-dsc/scripts/eval_gemma_vihallu.py --ckpt \"$BEST_CHECKPOINT\" --split test"
echo ""
echo "💡 Usage:"
echo "   Test mode:  . /root/uit-dsc/scripts/train_eval_loop.sh --test"
echo "   Full mode:  . /root/uit-dsc/scripts/train_eval_loop.sh"
