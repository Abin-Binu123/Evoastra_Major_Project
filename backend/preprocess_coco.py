import json
import os
import pandas as pd
from tqdm import tqdm

# Base COCO path
BASE_DIR = r"E:\Evoastra Internship\MAJOR PROJECT\data\coco"
ANNOTATION_FILE = os.path.join(BASE_DIR, "annotations", "captions_train2014.json")
IMAGE_DIR = os.path.join(BASE_DIR, "train2014")

# 1️⃣ Load COCO annotations
def load_captions(annotation_path):
    print("📥 Loading captions from:", annotation_path)
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    data = []
    for ann in tqdm(annotations['annotations'], desc="Extracting captions"):
        image_id = ann['image_id']
        caption = ann['caption'].strip()
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        data.append((image_filename, caption))
    return pd.DataFrame(data, columns=['image', 'caption'])

# 2️⃣ Verify if image exists
def verify_images(df, image_dir):
    print("\n🔍 Verifying images exist...")
    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(image_dir, row['image'])
        if os.path.exists(img_path):
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

# 3️⃣ Run preprocessing
if __name__ == "__main__":
    df = load_captions(ANNOTATION_FILE)
    print(f"\n✅ Total captions loaded: {len(df)}")

    df_valid = verify_images(df, IMAGE_DIR)
    print(f"✅ Verified {len(df_valid)} image-caption pairs exist")

    # Save to CSV
    output_csv = os.path.join(BASE_DIR, "captions_verified.csv")
    df_valid.to_csv(output_csv, index=False)
    print(f"\n💾 Saved verified captions to: {output_csv}")
