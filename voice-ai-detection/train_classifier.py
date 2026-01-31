import os
import torch
import librosa
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ----------------------------
# Device (CPU is enough)
# ----------------------------
device = "cpu"

# ----------------------------
# Load Wav2Vec2 (Frozen)
# ----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec.eval()  # freeze backbone

# ----------------------------
# Classifier Head
# ----------------------------
classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

# ----------------------------
# Helpers
# ----------------------------
def load_audio(path):
    audio, _ = librosa.load(path, sr=16000)
    return audio

def get_embedding(audio):
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        embeddings = wav2vec(**inputs).last_hidden_state
        pooled = embeddings.mean(dim=1)

    return pooled

# ----------------------------
# Training Logic
# ----------------------------
def train_epoch(folder, label):
    files = os.listdir(folder)
    total_loss = 0.0

    for f in files:
        path = os.path.join(folder, f)

        audio = load_audio(path)
        emb = get_embedding(audio)

        target = torch.tensor([[label]], dtype=torch.float32).to(device)
        pred = classifier(emb)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(files)

# ----------------------------
# Run Training
# ----------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    loss_human = train_epoch("data/human", 0)
    loss_ai = train_epoch("data/ai", 1)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Human loss: {loss_human:.4f} | "
        f"AI loss: {loss_ai:.4f}"
    )

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("model", exist_ok=True)
torch.save(classifier.state_dict(), "model/wav2vec_classifier.pt")

print("✅ Training complete.")
print("✅ Model saved to model/wav2vec_classifier.pt")
