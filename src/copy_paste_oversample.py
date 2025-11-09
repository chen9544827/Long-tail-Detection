from pathlib import Path
from PIL import Image
import random, os, shutil
ROOT = Path(__file__).parent.parent
imgs = ROOT / 'data' / 'train' / 'images'
lbls = ROOT / 'data' / 'train' / 'labels'
out = ROOT / 'data' / 'train_aug'
out_img = out / 'images'; out_lbl = out / 'labels'
out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

def read_boxes(p):
    out=[]
    with open(p,'r') as f:
        for l in f:
            a=l.split()
            if len(a)>=5:
                out.append((int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])))
    return out

# copy originals
for f in imgs.glob('*.*'):
    lbl = lbls / (f.stem + '.txt')
    if lbl.exists():
        shutil.copy2(f, out_img / f.name)
        shutil.copy2(lbl, out_lbl / lbl.name)

# collect minority crops (class_id list)
minority = [2]  # adjust
crops = []
for l in lbls.glob('*.txt'):
    boxes = read_boxes(l)
    for cls,cx,cy,w,h in boxes:
        if cls in minority:
            src = imgs / (l.stem + '.jpg')
            if not src.exists():
                src = imgs / (l.stem + '.png')
            if not src.exists():
                continue
            im = Image.open(src).convert('RGB')
            iw,ih = im.size
            bx = int((cx - w/2)*iw); by = int((cy - h/2)*ih)
            bw = int(w*iw); bh = int(h*ih)
            if bw<=0 or bh<=0: continue
            crop = im.crop((bx,by,bx+bw,by+bh))
            crops.append((crop, cls))
# paste crops onto random backgrounds
bg_list = list(out_img.glob('*.*'))
if len(bg_list)==0:
    bg_list = list(imgs.glob('*.*'))
for i,(crop,cls) in enumerate(crops):
    for rep in range(3):  # multiplier
        bg = Image.open(random.choice(bg_list)).convert('RGB')
        bw,bh = bg.size
        cw,ch = crop.size
        if cw>=bw or ch>=bh: continue
        x = random.randint(0, bw-cw); y = random.randint(0, bh-ch)
        bg.paste(crop, (x,y))
        name = f"cp_{i}_{rep}.jpg"
        bg.save(out_img / name)
        # write label (normalized)
        cx = (x + cw/2)/bw; cy = (y + ch/2)/bh; nw = cw/bw; nh = ch/bh
        with open(out_lbl / (name.replace('.jpg','.txt')), 'w') as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
print("copy-paste oversample done ->", out)