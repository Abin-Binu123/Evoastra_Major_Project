import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from collections import Counter
import nltk
import string

# Download NLTK tokenizer data (run once)
nltk.download("punkt")

# Paths
DATA_DIR = r"E:\Evoastra Internship\MAJOR PROJECT\data\coco"
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_verified.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train2014")

# 1️⃣ Vocabulary Builder
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Tokenize and clean punctuation
        return [word.lower() for word in nltk.word_tokenize(text) if word.isalpha()]

    def build_vocabulary(self, sentence_list):
        print("🧠 Building vocabulary...")
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print(f"✅ Vocabulary size: {len(self.itos)} words")

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]


# 2️⃣ Custom Dataset
class COCODataset(data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None, freq_threshold=5):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.image_dir, row["image"])
        caption = row["caption"]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numerical_caption)


# 3️⃣ Collate Function (for DataLoader)
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions


# 4️⃣ DataLoader builder
def get_loader(csv_file=CAPTIONS_FILE, image_dir=IMAGE_DIR, transform=None, batch_size=32, num_workers=2, shuffle=True, pin_memory=True):
    dataset = COCODataset(csv_file, image_dir, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset


# 5️⃣ Test run
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    loader, dataset = get_loader(transform=transform, batch_size=4)

    for idx, (imgs, captions) in enumerate(loader):
        print("Batch shape:", imgs.shape)
        print("Caption tensor shape:", captions.shape)
        break
