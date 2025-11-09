"""
創建平衡的 Fine-tune Dataset
策略: Under-sampling Head class + Over-sampling Tail class
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

def create_balanced_dataset(
    original_train_img='data/train/images',
    original_train_label='data/train/labels',
    output_dir='data/train_balanced',
    target_samples_per_class=3000,  # ⭐ 每個類別的目標樣本數
):
    """
    創建平衡的訓練集
    
    策略:
    - Class 0 (14421) → Under-sample 到 3000
    - Class 1 (647)   → Over-sample 到 3000 (複製 ~4.6x)
    - Class 2 (1924)  → Over-sample 到 3000 (複製 ~1.6x)
    - Class 3 (2854)  → Over-sample 到 3000 (複製 ~1.05x)
    """
    
    print("="*80)
    print("創建平衡的 Fine-tune Dataset")
    print("="*80)
    
    original_train_img = Path(original_train_img)
    original_train_label = Path(original_train_label)
    output_dir = Path(output_dir)
    
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 輸入:")
    print(f"   圖片: {original_train_img}")
    print(f"   標籤: {original_train_label}")
    print(f"\n📁 輸出:")
    print(f"   平衡集: {output_dir}")
    print(f"\n⚙️  策略:")
    print(f"   目標樣本數/類別: {target_samples_per_class}")
    
    # ⭐ Step 1: 分析每張圖片的類別分佈
    print(f"\n🔍 分析原始訓練集...")
    
    image_class_map = defaultdict(set)  # {image_name: {class_ids}}
    class_image_map = defaultdict(list)  # {class_id: [image_names]}
    
    for label_file in original_train_label.glob('*.txt'):
        img_name = label_file.stem
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    image_class_map[img_name].add(class_id)
                    class_image_map[class_id].append(img_name)
    
    # 統計
    print(f"\n📊 原始分佈 (圖片層級):")
    for class_id in sorted(class_image_map.keys()):
        count = len(set(class_image_map[class_id]))  # 去重
        print(f"   Class {class_id}: {count} 張圖片包含此類別")
    
    # ⭐ Step 2: 為每個類別選擇圖片
    print(f"\n🎯 建立平衡集...")
    
    selected_images = set()
    
    for class_id in sorted(class_image_map.keys()):
        class_images = list(set(class_image_map[class_id]))
        current_count = len(class_images)
        
        if current_count >= target_samples_per_class:
            # Under-sampling
            sampled = random.sample(class_images, target_samples_per_class)
            print(f"   Class {class_id}: Under-sample {current_count} → {target_samples_per_class}")
        else:
            # Over-sampling (允許重複)
            repeat_times = target_samples_per_class // current_count
            remainder = target_samples_per_class % current_count
            
            sampled = class_images * repeat_times
            sampled += random.sample(class_images, remainder)
            
            print(f"   Class {class_id}: Over-sample {current_count} → {target_samples_per_class} (重複 ~{repeat_times}x)")
        
        selected_images.update(sampled)
    
    # ⭐ Step 3: 複製檔案到新目錄
    print(f"\n📦 複製檔案...")
    
    copied_count = 0
    for img_name in selected_images:
        # 找到對應的圖片檔案
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = original_train_img / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"⚠️  找不到圖片: {img_name}")
            continue
        
        label_file = original_train_label / f"{img_name}.txt"
        
        if not label_file.exists():
            print(f"⚠️  找不到標籤: {img_name}.txt")
            continue
        
        # 複製
        shutil.copy(img_file, output_img_dir / img_file.name)
        shutil.copy(label_file, output_label_dir / label_file.name)
        copied_count += 1
    
    print(f"\n✅ 完成! 複製了 {copied_count} 張圖片")
    print(f"   輸出目錄: {output_dir}")
    
    # ⭐ Step 4: 創建新的 data.yaml
    create_balanced_yaml(output_dir)
    
    return output_dir


def create_balanced_yaml(balanced_dir):
    """創建新的 data.yaml 給平衡集"""
    
    yaml_content = {
        'path': str(Path.cwd()),
        'train': str(balanced_dir / 'images'),
        'val': 'data/val/images',  # ⭐ 驗證集不變
        'test': 'data/test/images',
        'nc': 4,
        'names': {
            0: 'class_0',
            1: 'class_1',
            2: 'class_2',
            3: 'class_3'
        }
    }
    
    yaml_path = Path('data_balanced.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n📝 創建新配置: {yaml_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=3000,
                        help='每個類別的目標樣本數')
    
    args = parser.parse_args()
    
    create_balanced_dataset(target_samples_per_class=args.target)