import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from backend.data_loader import get_loader
import os
import time

# =====================
# CONFIGURATION
# =====================
EPOCHS = 1                # Start small (increase later if needed)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_WORKERS = 0           # Keep 0 on Windows for stability
SUBSET_SIZE = 2000        # Use a subset for faster testing (set None for full dataset)
MODEL_SAVE_PATH = r"E:\Evoastra Internship\MAJOR PROJECT\models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# =====================
# MODEL DEFINITIONS
# =====================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]  # remove FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # BatchNorm fix for last small batch
        if features.size(0) > 1:
            features = self.bn(self.linear(features))
        else:
            features = self.linear(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs


# =====================
# TRAINING LOOP
# =====================

def train():
    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # Data loader
    loader, dataset = get_loader(transform=transform,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS)
    vocab_size = len(dataset.vocab)
    print(f"✅ Vocabulary size: {vocab_size}")

    # ⚙️ Optional: Train on a subset for faster testing
    if SUBSET_SIZE is not None and SUBSET_SIZE < len(loader.dataset.df):
        loader.dataset.df = loader.dataset.df.sample(SUBSET_SIZE).reset_index(drop=True)
        print(f"⚙️ Using subset of {SUBSET_SIZE} samples for faster training")

    encoder = EncoderCNN(EMBED_SIZE).to(device)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    print("\n🏋️ Starting training...\n")
    for epoch in range(EPOCHS):
        start_epoch_time = time.time()
        epoch_loss = 0

        for idx, (imgs, captions) in enumerate(loader):
            start_batch_time = time.time()
            imgs, captions = imgs.to(device), captions.to(device)

            # Forward pass
            features = encoder(imgs)
            outputs = decoder(features, captions)

            # Compute valid lengths
            lengths = (captions != dataset.vocab.stoi["<PAD>"]).sum(1)

            # Pack sequences (same length for outputs and targets)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]
            targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - start_batch_time

            # Log every 100 steps
            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f} | Batch Time: {batch_time:.2f}s")

        epoch_time = time.time() - start_epoch_time
        avg_loss = epoch_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1} completed in {epoch_time/60:.2f} mins | Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"💾 Model saved to {checkpoint_path}\n")


if __name__ == "__main__":
    train()
