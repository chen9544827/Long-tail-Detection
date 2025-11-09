import os
# 必須在 import torch / ultralytics / numpy 等之前設定
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 非安全但立即可用的 workaround
os.environ['OMP_NUM_THREADS'] = '1'           # 限制 OpenMP 執行緒數
os.environ['MKL_NUM_THREADS'] = '1'           # 限制 MKL 執行緒數
# ...existing code...
import sys
from pathlib import Path
import argparse
from shutil import copy2
from ultralytics import YOLO
import torch

# 確保可以導入同層級模組
sys.path.insert(0, str(Path(__file__).parent))
from config import config

def simple_oversample(train_images_dir: Path, train_labels_dir: Path, out_dir: Path, minority_classes: list, mult: int = 3):
    """
    簡單複製包含 minority_classes 的樣本到 out_dir (images, labels)
    不改變檔名結構，只做複製以增加出現頻率
    """
    out_images = out_dir / 'images'
    out_labels = out_dir / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 先複製全部原始檔案
    for img in train_images_dir.glob('*.*'):
        lbl = train_labels_dir / (img.stem + '.txt')
        if lbl.exists():
            copy2(img, out_images / img.name)
            copy2(lbl, out_labels / lbl.name)

    # 複製 minority 樣本
    for lbl in train_labels_dir.glob('*.txt'):
        with open(lbl, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        classes = [int(l.split()[0]) for l in lines] if lines else []
        if any(c in minority_classes for c in classes):
            src_img = None
            for ext in ['.png', '.jpg', '.jpeg']:
                p = train_images_dir / (lbl.stem + ext)
                if p.exists():
                    src_img = p
                    break
            if not src_img:
                continue
            for i in range(mult):
                dst_img = out_images / f"{lbl.stem}_dup{i}{src_img.suffix}"
                dst_lbl = out_labels / f"{lbl.stem}_dup{i}.txt"
                copy2(src_img, dst_img)
                copy2(lbl, dst_lbl)
    return out_dir

def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 Fine-tune Script')
    p.add_argument('--weights', required=True, help='已訓練的權重檔 (best.pt)')
    p.add_argument('--data', default='./data.yaml', help='data.yaml 路徑 (相對於 src)')
    p.add_argument('--output', default='./runs/detect/yolov8m_finetune', help='輸出目錄')
    p.add_argument('--epochs', type=int, default=30, help='微調 epochs')
    p.add_argument('--imgsz', type=int, default=896, help='image size')
    p.add_argument('--batch', type=int, default=8, help='batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='初始學習率 (lr0)')
    p.add_argument('--box', type=float, default=8.5, help='box loss gain')
    p.add_argument('--dfl', type=float, default=2.0, help='dfl loss gain')
    p.add_argument('--oversample', action='store_true', help='啟用簡單 oversample (copy minority)')
    p.add_argument('--mult', type=int, default=3, help='oversample multiplier')
    p.add_argument('--minority', type=int, nargs='*', default=[2], help='要增樣的 class id (例如 2)')
    p.add_argument('--device', default=None, help='cuda device or cpu')
    return p.parse_args()

def main():
    args = parse_args()
    src_root = Path(__file__).parent.parent
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到權重: {weights_path}")

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        data_yaml = src_root / args.data
        if not data_yaml.exists():
            raise FileNotFoundError(f"找不到 data.yaml: {args.data}")

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 可選 oversample（會建立 data/train_aug 資料夾）
    train_dir_override = None
    if args.oversample:
        print("執行簡單 oversample ...")
        # 假設 data.yaml 指向 data/train/images 與 data/train/labels（YOLO 標準）
        # 解析 data.yaml 簡單取 train 目錄
        import yaml
        with open(data_yaml, 'r') as f:
            d = yaml.safe_load(f)
        train_path = Path(d.get('train'))
        if not train_path.is_dir():
            # 支援相對路徑
            train_path = src_root / d.get('train')
        images_dir = train_path / 'images'
        labels_dir = train_path / 'labels'
        out_dir = src_root / 'data' / 'train_aug'
        simple_oversample(images_dir, labels_dir, out_dir, args.minority, args.mult)
        # 修改臨時 data.yaml 指向 train_aug
        import copy
        new_data = copy.deepcopy(d)
        new_data['train'] = str(out_dir)
        temp_data_yaml = src_root / 'data' / 'data_finetune.yaml'
        with open(temp_data_yaml, 'w') as f:
            yaml.safe_dump(new_data, f)
        data_yaml = temp_data_yaml
        print(f"已建立 oversample 資料: {out_dir}")
        train_dir_override = out_dir

    print("載入模型...")
    model = YOLO(str(weights_path))

    print("開始 fine-tuning ...")
    save_project = Path(args.output).resolve()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer='AdamW',
        lr0=args.lr,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=1.0,
        # 較保守增強
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.3,
        translate=0.06,
        scale=0.25,
        mosaic=0.6,
        mixup=0.12,
        copy_paste=0.25,
        # loss
        box=args.box,
        cls=0.5,
        dfl=args.dfl,
        # output
        project=str(save_project.parent),
        name=str(save_project.name),
        exist_ok=True,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        pretrained=False,  # 使用現有 weights 做微調
        amp=True,
        deterministic=True,
        verbose=True,
    )

    print("Fine-tune 完成, 開始驗證...")
    best_weights = results.save_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        val_results = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=0.001,
            iou=0.6,
            device=device,
            plots=True,
            save_json=True,
        )
        print(f"驗證完成: mAP@0.5 {val_results.box.map50:.4f}, mAP@0.5:0.95 {val_results.box.map:.4f}")
    else:
        print(f"找不到 fine-tune 輸出 best.pt (預期於 {results.save_dir / 'weights'} )")

if __name__ == '__main__':
    main()