"""
診斷腳本 - 分析 submission.csv 的空預測情況
"""
import csv
import sys
from pathlib import Path

def analyze_submission(csv_path):
    """分析提交檔案"""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ 找不到檔案: {csv_path}")
        return
    
    print("="*80)
    print("分析 Kaggle Submission 檔案")
    print("="*80)
    print(f"\n📄 檔案: {csv_path}")
    
    total_rows = 0
    empty_predictions = 0
    prediction_lengths = []
    sample_empty = []
    sample_filled = []
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        print(f"\n📋 Header: {header}")
        
        for row in reader:
            total_rows += 1
            if len(row) < 2:
                print(f"⚠️  警告: 第 {total_rows+1} 行格式錯誤: {row}")
                continue
            
            img_id, pred_str = row[0], row[1]
            pred_str = pred_str.strip()
            
            if pred_str == '':
                empty_predictions += 1
                if len(sample_empty) < 5:
                    sample_empty.append(img_id)
            else:
                # 計算預測框數量 (每個框有 6 個值: conf x y w h class)
                parts = pred_str.split()
                num_boxes = len(parts) // 6
                prediction_lengths.append(num_boxes)
                if len(sample_filled) < 5:
                    sample_filled.append((img_id, num_boxes, pred_str[:100]))
    
    print(f"\n📊 統計結果:")
    print(f"   總行數: {total_rows}")
    print(f"   有預測: {total_rows - empty_predictions} ({(total_rows-empty_predictions)/total_rows*100:.1f}%)")
    print(f"   空預測: {empty_predictions} ({empty_predictions/total_rows*100:.1f}%)")
    
    if prediction_lengths:
        avg_boxes = sum(prediction_lengths) / len(prediction_lengths)
        max_boxes = max(prediction_lengths)
        min_boxes = min(prediction_lengths)
        print(f"\n📦 檢測框統計 (非空圖片):")
        print(f"   平均: {avg_boxes:.1f} 框/圖")
        print(f"   最大: {max_boxes} 框")
        print(f"   最小: {min_boxes} 框")
    
    if sample_empty:
        print(f"\n🔍 空預測範例 (前 5 個 Image_ID):")
        for img_id in sample_empty:
            print(f"   {img_id}")
    
    if sample_filled:
        print(f"\n✅ 有預測範例 (前 5 個):")
        for img_id, num, snippet in sample_filled:
            print(f"   {img_id}: {num} 框")
            print(f"      {snippet}...")
    
    # 判斷
    print(f"\n💡 診斷:")
    if empty_predictions == 0:
        print("   ✅ 太好了！沒有任何空預測")
    elif empty_predictions / total_rows < 0.05:
        print(f"   ✅ 空預測比例很低 ({empty_predictions/total_rows*100:.1f}%)，這是正常的")
    elif empty_predictions / total_rows < 0.15:
        print(f"   ⚠️  空預測比例偏高 ({empty_predictions/total_rows*100:.1f}%)")
        print("   建議:")
        print("   1. 降低信心度閾值 (--conf 0.001)")
        print("   2. 使用 inference_yolo_aggressive.py 激進推論")
    else:
        print(f"   ❌ 空預測比例過高 ({empty_predictions/total_rows*100:.1f}%)!")
        print("   強烈建議:")
        print("   1. 使用激進推論腳本 (inference_yolo_aggressive.py)")
        print("   2. 啟用多尺度 (--multiscale) 和 TTA (--tta)")
        print("   3. 檢查模型是否訓練不足或過擬合")
    
    return {
        'total': total_rows,
        'empty': empty_predictions,
        'empty_ratio': empty_predictions / total_rows if total_rows > 0 else 0
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 預設路徑
        csv_path = Path(__file__).parent.parent / 'kaggle_submission_final' / 'submission.csv'
        if not csv_path.exists():
            csv_path = Path(__file__).parent.parent / 'kaggle_submission' / 'submission.csv'
    
    result = analyze_submission(csv_path)
    
    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}")
