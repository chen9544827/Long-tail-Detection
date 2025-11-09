class Config:
    def __init__(self):
        # ============================================================
        # 資料集配置
        # ============================================================
        self.NUM_CLASSES = 4  # Class 0, 1, 2, 3
        self.CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3']
        
        # 類別分佈 (用於 loss weighting)
        self.CLASS_DISTRIBUTION = {
            0: 14421,  # 72.66% (Head)
            1: 647,    # 3.26%  (Long Tail)
            2: 1924,   # 9.69%  (Tail)
            3: 2854    # 14.38% (Medium)
        }
        
        # ============================================================
        # 輸入配置
        # ============================================================
        self.INPUT_SIZE = (640, 640)  # 先用 640,後續可提高到 1280
        self.INPUT_CHANNELS = 3
        
        # ============================================================
        # Anchor 配置 (基於 K-means 分析)
        # ============================================================
        self.ANCHORS = [
            # 小尺度 (stride=8) - 主要檢測 Class 2 (極小), Class 3
            [[14, 15], [23, 31], [40, 21]],
            
            # 中尺度 (stride=16) - 主要檢測 Class 0, Class 3
            [[29, 55], [55, 31], [67, 42]],
            
            # 大尺度 (stride=32) - 主要檢測 Class 0, Class 1 (最大)
            [[42, 67], [112, 57], [128, 105]]
        ]
        
        self.STRIDES = [8, 16, 32]
        self.NUM_ANCHORS = 3  # 每個尺度 3 個 anchors
        
        # ============================================================
        # 網路架構
        # ============================================================
        self.BACKBONE = 'resnet50'  # 或 'resnet101', 'efficientnet'
        self.PRETRAINED = True
        self.FREEZE_BACKBONE = False  # 不凍結,允許 fine-tune
        
        # FPN 配置
        self.FPN_CHANNELS = 256
        
        # ============================================================
        # 訓練配置
        # ============================================================
        # 基礎參數
        self.BATCH_SIZE = 16  # 根據 GPU 記憶體調整
        self.NUM_EPOCHS = 20
        self.NUM_WORKERS = 2
        
        # Learning rate
        self.LEARNING_RATE = 0.005
        self.WEIGHT_DECAY = 5e-4
        self.MOMENTUM = 0.9
        
        # LR Scheduler
        self.LR_SCHEDULER = 'step'  # 'step', 'cosine', 'warmup_cosine'
        self.WARMUP_EPOCHS = 5
        self.LR_DECAY_EPOCHS = [15, 18]
        self.LR_DECAY_GAMMA = 0.1
        
        # ============================================================
        # Loss 配置 (針對 Long-Tail)
        # ============================================================
        # Loss 類型
        self.USE_FOCAL_LOSS = True  # Focal Loss 處理類別不平衡
        self.FOCAL_ALPHA = [1.0, 22.3, 7.5, 5.1]  # 基於類別比例 (head/class)
        self.FOCAL_GAMMA = 2.0
        
        self.USE_GIOU_LOSS = True  # GIoU 改善小物體定位
        
        # Loss 權重
        self.LAMBDA_COORD = 5.0   # 定位損失 (提高,因為有大量小物體)
        self.LAMBDA_OBJ = 1.0     # 物體存在損失
        self.LAMBDA_NOOBJ = 0.5   # 背景損失 (降低,減少假陽性)
        self.LAMBDA_CLASS = 2.0   # 分類損失 (提高,處理 Long-Tail)
        
        # ============================================================
        # 資料增強 (針對 Long-Tail)
        # ============================================================
        # 基礎增強
        self.AUGMENTATION = True
        self.MOSAIC = True  # Mosaic 增強 (YOLO v4/v5)
        self.MOSAIC_PROB = 0.3
        
        self.MIXUP = True  # MixUp 增強
        self.MIXUP_PROB = 0.1
        
        # 顏色增強
        self.COLOR_JITTER = True
        self.BRIGHTNESS = 0.2
        self.CONTRAST = 0.2
        self.SATURATION = 0.2
        self.HUE = 0.1
        
        # 幾何增強
        self.RANDOM_FLIP = True
        self.FLIP_PROB = 0.5
        
        self.RANDOM_SCALE = True
        self.SCALE_RANGE = (0.8, 1.2)
        
        self.RANDOM_CROP = True
        self.CROP_PROB = 0.3
        
        # 針對稀有類別的特殊增強
        self.TAIL_CLASS_AUGMENTATION = True
        self.TAIL_CLASSES = [1, 2]  # Class 1, 2 需要額外增強
        self.TAIL_AUG_REPEAT = 2  # 稀有類別樣本重複次數
        
        # ============================================================
        # Re-sampling 配置 (針對 Long-Tail)
        # ============================================================
        self.USE_RESAMPLING = True
        self.SAMPLING_STRATEGY = 'square_root'  # 'inverse', 'square_root', 'class_balanced'
        
        # ============================================================
        # 推論配置
        # ============================================================
        self.CONF_THRESHOLD = 0.25  # 置信度閾值
        self.NMS_THRESHOLD = 0.45   # NMS 閾值 (密集物體需較寬鬆)
        self.MAX_DETECTIONS = 200   # 每張圖最多檢測數 (平均 35 個,給予餘裕)
        
        # 多尺度測試
        self.MULTI_SCALE_TEST = False  # 推論時使用多尺度
        self.TEST_SCALES = [0.8, 1.0, 1.2]
        
        # Test-Time Augmentation
        self.TEST_TIME_AUG = False
        
        # ============================================================
        # 其他配置
        # ============================================================
        # 路徑配置 (相對於專案根目錄)
        import os
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.SAVE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
        self.LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
        self.VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
        
        self.SAVE_INTERVAL = 5  # 每 N epochs 儲存一次
        self.EVAL_INTERVAL = 2  # 每 N epochs 驗證一次
        self.VIS_INTERVAL = 5   # ⭐ 新增: 每 N 個 epoch 視覺化一次
        
        self.DEVICE = 'cuda'  # 'cuda' or 'cpu'
        self.SEED = 42
        
    def get_class_weights(self):
        """計算類別權重 (用於 Focal Loss alpha)"""
        total = sum(self.CLASS_DISTRIBUTION.values())
        max_count = max(self.CLASS_DISTRIBUTION.values())
        
        weights = []
        for class_id in sorted(self.CLASS_DISTRIBUTION.keys()):
            count = self.CLASS_DISTRIBUTION[class_id]
            # 使用 inverse frequency
            weight = max_count / count
            weights.append(weight)
        
        return weights
    
    def get_sampling_weights(self):
        """計算採樣權重 (用於 Re-sampling)"""
        import numpy as np
        
        counts = [self.CLASS_DISTRIBUTION[i] for i in sorted(self.CLASS_DISTRIBUTION.keys())]
        
        if self.SAMPLING_STRATEGY == 'inverse':
            weights = 1.0 / np.array(counts)
        elif self.SAMPLING_STRATEGY == 'square_root':
            weights = 1.0 / np.sqrt(counts)
        else:  # class_balanced
            weights = np.ones(len(counts))
        
        # 正規化
        weights = weights / weights.sum()
        
        return {i: w for i, w in enumerate(weights)}
    
    def print_config(self):
        """列印配置摘要"""
        print("="*60)
        print("Configuration Summary")
        print("="*60)
        print(f"\n[Dataset]")
        print(f"  Classes: {self.NUM_CLASSES}")
        print(f"  Input Size: {self.INPUT_SIZE}")
        
        print(f"\n[Network]")
        print(f"  Backbone: {self.BACKBONE}")
        print(f"  Anchors: {len(self.ANCHORS)} scales × {self.NUM_ANCHORS} anchors")
        
        print(f"\n[Training]")
        print(f"  Batch Size: {self.BATCH_SIZE}")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Use Focal Loss: {self.USE_FOCAL_LOSS}")
        print(f"  Use Re-sampling: {self.USE_RESAMPLING}")
        
        print(f"\n[Long-Tail Strategy]")
        print(f"  Focal Alpha: {self.FOCAL_ALPHA}")
        print(f"  Sampling Strategy: {self.SAMPLING_STRATEGY}")
        print(f"  Tail Class Aug: {self.TAIL_CLASS_AUGMENTATION}")
        
        print("="*60)

# 建立全域 config 實例
config = Config()

if __name__ == "__main__":
    config.print_config()
    
    print("\n[Class Weights]")
    weights = config.get_class_weights()
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.2f}")
    
    print("\n[Sampling Weights]")
    sampling_weights = config.get_sampling_weights()
    for cls, w in sampling_weights.items():
        print(f"  Class {cls}: {w:.4f}")