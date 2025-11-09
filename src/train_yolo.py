"""
YOLOv8 訓練腳本 - Long-Tailed Object Detection
針對類別不平衡問題使用特殊訓練策略
"""
import os
# ⭐⭐⭐ 必須在最開頭!在 import 任何其他庫之前 ⭐⭐⭐
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

# 確保可以導入同層級的模組
sys.path.insert(0, str(Path(__file__).parent))

from config import config


def create_class_weights():
    """根據類別分佈計算權重"""
    class_dist = config.CLASS_DISTRIBUTION
    total = sum(class_dist.values())
    max_count = max(class_dist.values())
    
    # 使用 inverse frequency 計算權重
    weights = []
    for class_id in sorted(class_dist.keys()):
        count = class_dist[class_id]
        weight = max_count / count  # 頭部類別權重小,尾部類別權重大
        weights.append(weight)
    
    # 標準化權重
    total_weight = sum(weights)
    weights = [w / total_weight * len(weights) for w in weights]
    
    return weights


def train_yolo():
    """訓練 YOLOv8 模型 - 優化版 v2"""
    
    print("="*80)
    print(" "*20 + "YOLOv8 Long-Tailed Object Detection v2")
    print(" "*25 + "(優化超參數)")
    print("="*80)
    
    # 檢查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  使用裝置: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 專案路徑
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / 'data.yaml'
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到資料集配置: {data_yaml}")
    
    print(f"\n📁 資料集配置: {data_yaml}")
    
    # 載入模型
    model_size = 'yolov8m'
    print(f"\n🤖 載入模型: {model_size}.pt")
    model = YOLO(f'{model_size}.pt')
    
    print("\n" + "="*80)
    print("🚀 開始訓練 (優化配置)")
    print("="*80)
    
    # ============================================================
    # ⭐⭐⭐ 關鍵改進點 ⭐⭐⭐
    # ============================================================
    
    results = model.train(
        # ============================================================
        # 基礎配置
        # ============================================================
        data=str(data_yaml),
        epochs=100,                     # ⭐ 150 → 100 (防止過擬合)
        imgsz=896,                      
        batch=12,                        # ⭐ 如果 VRAM 足夠可改為 12-16
        device=device,
        workers=8,
        
        # ============================================================
        # 優化器配置 (關鍵改進)
        # ============================================================
        optimizer='AdamW',              # ⭐ SGD → AdamW (更穩定)
        lr0=0.002,                      # ⭐ 0.003 → 0.002 (降低學習率)
        lrf=0.001,                      # ⭐ 0.01 → 0.001 (更平滑衰減)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,              # ⭐ 3.0 → 5.0 (更長熱身)
        warmup_momentum=0.8,
        warmup_bias_lr=0.0001,
        
        # ============================================================
        # 資料增強 (加強 Long-Tail 處理)
        # ============================================================
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,                   # ⭐ 0.0 → 10.0 (加入旋轉)
        translate=0.15,                 # ⭐ 0.1 → 0.15 (加強平移)
        scale=0.5,                      # ⭐ 0.25 → 0.5 (加強縮放)
        shear=2.0,                      # ⭐ 0.0 → 2.0 (加入剪切)
        perspective=0.0001,             # ⭐ 加入輕微透視
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,                     # ⭐ 0.6 → 1.0 (提高 mosaic)
        mixup=0.2,                      # ⭐ 0.1 → 0.2 (提高 mixup)
        copy_paste=0.5,                 # ⭐ 0.25 → 0.5 (大幅提高)
        
        # ============================================================
        # Loss 配置 (針對小物體和類別不平衡)
        # ============================================================
        box=9.0,                        # ⭐ 8.0 → 9.0 (更重視框準確度)
        cls=0.6,                        # ⭐ 0.5 → 0.6 (提高分類權重)
        dfl=2.0,                        # ⭐ 1.5 → 2.0 (提高分佈焦點)
        
        # ⭐ YOLOv8 不直接支援 class_weights,但可透過 cls loss 調整
        
        # ============================================================
        # 訓練策略
        # ============================================================
        patience=25,                    # ⭐ 50 → 25 (更早停止)
        save=True,
        save_period=5,
        
        # ============================================================
        # 驗證配置
        # ============================================================
        val=True,
        plots=True,
        
        # ============================================================
        # 輸出配置
        # ============================================================
        project=str(project_root / 'runs' / 'detect'),
        name='yolov8m_optimized_v3',    # ⭐ 新名稱
        exist_ok=False,                 # ⭐ True → False (避免覆蓋)
        
        # ============================================================
        # 其他配置
        # ============================================================
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=False,            # ⭐ True → False (更快)
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=15,                # ⭐ 10 → 15 (更早關閉)
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # ============================================================
        # 進階配置
        # ============================================================
        cache=False,                    # ⭐ 如果 RAM 足夠可設為 True
        label_smoothing=0.0,
    )
    
    print("\n" + "="*80)
    print("🎊 訓練完成!")
    print("="*80)
    
    # 顯示結果
    print(f"\n📊 訓練結果:")
    print(f"   最佳模型: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"   最後模型: {results.save_dir / 'weights' / 'last.pt'}")
    
    return results


def validate_yolo(weights_path):
    """驗證訓練好的模型"""
    print("\n" + "="*80)
    print("驗證模型")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / 'data.yaml'
    
    model = YOLO(weights_path)
    
    results = model.val(
        data=str(data_yaml),
        imgsz=896,
        batch=8,
        conf=0.001,
        iou=0.6,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        plots=True,
        save_json=True,
        save_hybrid=True,
    )
    
    print(f"\n" + "="*80)
    print("📊 驗證結果")
    print("="*80)
    
    print(f"\n整體指標:")
    print(f"  mAP@0.5     : {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f} ⭐")
    print(f"  Precision   : {results.box.mp:.4f}")
    print(f"  Recall      : {results.box.mr:.4f}")
    
    # 顯示各類別 AP
    print(f"\n各類別 AP@0.5:")
    for i, ap in enumerate(results.box.ap50):
        count = config.CLASS_DISTRIBUTION[i]
        pct = count / sum(config.CLASS_DISTRIBUTION.values()) * 100
        print(f"  {config.CLASS_NAMES[i]:10s}: {ap:.4f} (樣本數: {count:5d}, {pct:5.2f}%)")
    
    print(f"\n各類別 AP@0.5:0.95:")
    for i, ap in enumerate(results.box.ap):
        print(f"  {config.CLASS_NAMES[i]:10s}: {ap:.4f}")
    
    return results

def finetune_best_model():
    """
    從 best.pt Fine-tune
    使用平衡的 dataset (方案 A)
    
    策略:
    1. 使用 data_balanced.yaml (平衡訓練集)
    2. 極低學習率 (避免破壞原有性能)
    3. 凍結前面層 (只調整後面層)
    4. 輕度資料增強
    5. 不使用額外的 class weights (dataset 已平衡)
    """
    print("="*80)
    print(" "*15 + "Fine-tune Best Model with Balanced Dataset")
    print(" "*25 + "(方案 A)")
    print("="*80)
    
    # ========== 路徑設定 ==========
    project_root = Path(__file__).parent.parent
    best_model_path = project_root / "runs/detect/yolov8_longtail/weights/best.pt"
    balanced_yaml = project_root / "data_balanced.yaml"
    
    # 檢查模型
    if not best_model_path.exists():
        print(f"❌ 找不到模型: {best_model_path}")
        print(f"   請確認路徑是否正確")
        return
    
    # 檢查平衡 dataset
    if not balanced_yaml.exists():
        print(f"❌ 找不到平衡 dataset 配置: {balanced_yaml}")
        print(f"\n請先執行:")
        print(f"   python create_balanced_dataset.py --target 3000")
        return
    
    print(f"\n📦 模型路徑: {best_model_path}")
    print(f"📊 Dataset: {balanced_yaml} (平衡訓練集)")
    
    # ========== 載入模型 ==========
    print(f"\n🔧 載入模型...")
    model = YOLO(str(best_model_path))
    
    # 檢查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用裝置: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== Fine-tune 策略說明 ==========
    print(f"\n{'='*80}")
    print("⚙️  Fine-tune 配置 (平衡 Dataset 策略)")
    print(f"{'='*80}")
    
    print(f"\n📚 Dataset:")
    print(f"   訓練集: data/train_balanced (每類 ~3000 樣本)")
    print(f"   驗證集: data/val (原始驗證集,不變)")
    
    print(f"\n🎯 訓練策略:")
    print(f"   Epochs: 40 (中期訓練)")
    print(f"   Batch Size: 16")
    print(f"   Image Size: 896")
    
    print(f"\n📉 學習率:")
    print(f"   初始學習率: 0.0001 (極低,保護原有權重)")
    print(f"   最終學習率: 0.00001")
    print(f"   Scheduler: Cosine Annealing")
    
    print(f"\n🔒 模型凍結:")
    print(f"   凍結層: 前 10 層 (只訓練後面的檢測頭)")
    print(f"   原因: 前面層已學會基礎特徵,不需要重新訓練")
    
    print(f"\n🎨 資料增強 (輕度):")
    print(f"   HSV: 輕度調整 (h=0.01, s=0.3, v=0.3)")
    print(f"   幾何: 輕度變換 (rotate=5°, translate=5%)")
    print(f"   Mosaic: 0.7 (降低複雜度)")
    print(f"   Mixup: 0.0 (不使用)")
    
    print(f"\n⚖️  類別權重:")
    print(f"   不使用額外權重 (dataset 已平衡)")
    
    print(f"\n⏰ Early Stopping:")
    print(f"   Patience: 20 epochs")
    print(f"   目標: 防止過擬合")
    
    # ========== 開始訓練 ==========
    print(f"\n{'='*80}")
    print("🚀 開始 Fine-tuning...")
    print(f"{'='*80}\n")
    
    try:
        results = model.train(
            # ========== Dataset ==========
            data=str(balanced_yaml),  # ⭐ 使用平衡 dataset
            
            # ========== 基本設定 ==========
            epochs=40,                # 中期訓練
            imgsz=896,                # 與原訓練一致
            batch=16,                 # 根據 VRAM 調整
            device=device,
            workers=8,
            
            # ========== 學習率 (極低) ==========
            lr0=0.0001,              # ⭐ 初始學習率 (原本 0.01 的 1/100)
            lrf=0.00001,             # ⭐ 最終學習率
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,         # 熱身 3 epochs
            warmup_momentum=0.8,
            warmup_bias_lr=0.0001,
            
            # ========== 凍結層 ==========
            freeze=10,               # ⭐ 凍結前 10 層
            
            # ========== 資料增強 (輕度) ==========
            hsv_h=0.01,              # 色調 (降低)
            hsv_s=0.3,               # 飽和度 (降低)
            hsv_v=0.3,               # 明度 (降低)
            degrees=5,               # 旋轉角度 (降低)
            translate=0.05,          # 平移 (降低)
            scale=0.3,               # 縮放 (降低)
            shear=0.0,               # 不使用剪切
            perspective=0.0,         # 不使用透視
            flipud=0.0,              # 不使用上下翻轉
            fliplr=0.5,              # 保持左右翻轉
            mosaic=0.7,              # 降低 mosaic (原本 1.0)
            mixup=0.0,               # ⭐ 不使用 mixup
            copy_paste=0.0,          # 不使用 copy-paste
            
            # ========== 優化器 ==========
            optimizer='AdamW',       # AdamW 對 Fine-tune 更好
            
            # ========== Early Stopping ==========
            patience=20,             # 20 epochs 沒改善就停止
            
            # ========== 儲存設定 ==========
            save=True,
            save_period=2,           # 每 2 epochs 儲存
            
            # ========== 專案設定 ==========
            project=str(project_root / 'runs/detect'),
            name='finetune_balanced_v1',  # ⭐ 專案名稱
            exist_ok=False,
            pretrained=False,        # ⭐ 不載入預訓練 (已用 best.pt)
            
            # ========== 其他 ==========
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,              # 不使用矩形訓練
            cos_lr=True,             # ⭐ Cosine Learning Rate
            close_mosaic=10,         # 最後 10 epochs 關閉 mosaic
            amp=True,                # 自動混合精度 (加速訓練)
            fraction=1.0,            # 使用 100% 資料
            
            # ========== Validation ==========
            val=True,
            plots=True,
            
            # ========== Loss 權重 (可選) ==========
            box=7.5,                 # Box loss 權重
            cls=0.5,                 # Class loss 權重
            dfl=1.5,                 # DFL loss 權重
        )
        
        # ========== 訓練完成 ==========
        print("\n" + "="*80)
        print("✅ Fine-tuning 完成!")
        print("="*80)
        
        # 顯示結果路徑
        print(f"\n📊 訓練結果:")
        print(f"   專案目錄: {results.save_dir}")
        print(f"   最佳權重: {results.save_dir}/weights/best.pt")
        print(f"   最後權重: {results.save_dir}/weights/last.pt")
        print(f"   訓練曲線: {results.save_dir}/results.png")
        print(f"   CSV 結果: {results.save_dir}/results.csv")
        
        # ========== 自動驗證最佳模型 ==========
        print(f"\n{'='*80}")
        print("🔍 驗證最佳模型...")
        print(f"{'='*80}")
        
        best_path = Path(results.save_dir) / 'weights/best.pt'
        
        if best_path.exists():
            # 在原始驗證集上驗證
            val_model = YOLO(str(best_path))
            val_results = val_model.val(data=str(project_root / 'data.yaml'))
            
            print(f"\n📈 驗證集表現 (原始驗證集):")
            print(f"   mAP@0.5     : {val_results.box.map50:.5f}")
            print(f"   mAP@0.5:0.95: {val_results.box.map:.5f}")
            print(f"   Precision   : {val_results.box.p[0]:.5f}")
            print(f"   Recall      : {val_results.box.r[0]:.5f}")
            
            # 比較與原始 best.pt
            print(f"\n📊 與原始模型比較:")
            print(f"   原始 best.pt mAP@0.5:0.95: ~0.19094")
            print(f"   Fine-tuned   mAP@0.5:0.95: {val_results.box.map:.5f}")
            
            improvement = val_results.box.map - 0.19094
            if improvement > 0:
                print(f"   ✅ 改善: +{improvement:.5f} ({improvement/0.19094*100:.2f}%)")
            else:
                print(f"   ⚠️  下降: {improvement:.5f} ({improvement/0.19094*100:.2f}%)")
        
        # ========== 下一步建議 ==========
        print(f"\n{'='*80}")
        print("📌 下一步:")
        print(f"{'='*80}")
        print(f"\n1. 檢查訓練曲線:")
        print(f"   start {results.save_dir}/results.png")
        
        print(f"\n2. 使用 Fine-tuned 模型推論:")
        print(f"   python inference_yolo.py \\")
        print(f"     --weights \"{best_path}\" \\")
        print(f"     --image_dir \"../data/test/img\" \\")
        print(f"     --output_dir \"../sub_finetuned\" \\")
        print(f"     --conf 0.03 --no_vis")
        
        print(f"\n3. 提交到 Kaggle 比較分數")
        
        print(f"\n4. 如果效果不佳,可以:")
        print(f"   - 降低學習率 (lr0=0.00005)")
        print(f"   - 凍結更多層 (freeze=15)")
        print(f"   - 減少 epochs (epochs=30)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 訓練過程發生錯誤:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函數 - 處理不同模式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YOLOv8 訓練/驗證/Fine-tune 腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 訓練
  python train_yolo.py --mode train
  
  # 驗證
  python train_yolo.py --mode val --weights "runs/detect/xxx/weights/best.pt"
  
  # Fine-tune
  python train_yolo.py --mode finetune
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'val', 'finetune'],
        help='運行模式: train(訓練) / val(驗證) / finetune(微調)'
    )
    
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None,
        help='驗證模式時的權重路徑'
    )
    
    args = parser.parse_args()
    
    # 顯示模式
    print("\n" + "="*80)
    print(f"模式: {args.mode.upper()}")
    print("="*80 + "\n")
    
    # 根據模式執行
    if args.mode == 'train':
        train_yolo()
        
    elif args.mode == 'val':
        if args.weights is None:
            print("❌ 錯誤: 驗證模式需要 --weights 參數")
            print("\n使用範例:")
            print('  python train_yolo.py --mode val --weights "runs/detect/xxx/weights/best.pt"')
            return
        validate_yolo(args.weights)
        
    elif args.mode == 'finetune':
        finetune_best_model()
        
    else:
        print(f"❌ 錯誤: 未知模式 '{args.mode}'")
        print(f"   可用模式: train, val, finetune")


if __name__ == "__main__":
    # 確保導入必要的庫
    import sys
    import os
    import torch
    from pathlib import Path
    from ultralytics import YOLO
    
    # 執行主函數
    main()
