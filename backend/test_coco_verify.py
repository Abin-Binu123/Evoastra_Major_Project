import os
import json

# Path to your COCO dataset
BASE_PATH = r"E:\Evoastra Internship\MAJOR PROJECT\data\coco"

train_dir = os.path.join(BASE_PATH, "train2014")
val_dir = os.path.join(BASE_PATH, "val2014")
annotation_dir = os.path.join(BASE_PATH, "annotations")

# Check image folders
print("✅ Checking image folders...\n")
print("Train images folder exists:", os.path.exists(train_dir))
print("Val images folder exists:", os.path.exists(val_dir))

# Count sample images
if os.path.exists(train_dir):
    print("Total train images:", len(os.listdir(train_dir)))
if os.path.exists(val_dir):
    print("Total val images:", len(os.listdir(val_dir)))

# Check annotation files
print("\n✅ Checking annotation files...\n")
train_json = os.path.join(annotation_dir, "captions_train2014.json")
val_json = os.path.join(annotation_dir, "captions_val2014.json")

print("Train captions JSON exists:", os.path.exists(train_json))
print("Val captions JSON exists:", os.path.exists(val_json))

# Read small sample from JSON
if os.path.exists(train_json):
    with open(train_json, "r") as f:
        data = json.load(f)
    print("\nSample caption entry:")
    print(data["annotations"][0])
