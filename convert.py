import os
import json
import shutil
import random

# paths

base_dir = os.path.dirname(os.path.abspath(__file__))
coco_json = os.path.join(base_dir, "raw_data", "NM_NGD_coco_dataset", "train", "annotations.json")
images_dir = os.path.join(base_dir, "raw_data", "NM_NGD_coco_dataset", "train", "images")
output_dir = os.path.join(base_dir, "dataset")

os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

# load json

with open(coco_json) as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

# split train/val

image_ids = list(images.keys())
rng = random.Random(42)
rng.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.8)
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]

# group annotations by image

ann_map = {}
for ann in data["annotations"]:
    ann_map.setdefault(ann["image_id"], []).append(ann)


def convert_bbox(size, bbox):
    w, h = size
    x, y, bw, bh = bbox
    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    bw /= w
    bh /= h
    return x_center, y_center, bw, bh

def process(ids, split):
    for img_id in ids:
        img = images[img_id]
        filename = img["file_name"]
        width, height = img["width"], img["height"]

        src = os.path.join(images_dir, filename.replace("/", os.sep))
        dst = os.path.join(output_dir, f"images/{split}", filename)

        if not os.path.exists(src):
            continue

        # copy image
        shutil.copy2(src, dst)

        stem, _ = os.path.splitext(filename)
        label_path = os.path.join(output_dir, f"labels/{split}", f"{stem}.txt")

        with open(label_path, "w") as f:
            for ann in ann_map.get(img_id, []):
                cls = ann["category_id"]
                bbox = convert_bbox((width, height), ann["bbox"])
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

process(train_ids, "train")
process(val_ids, "val")

print("Done!")