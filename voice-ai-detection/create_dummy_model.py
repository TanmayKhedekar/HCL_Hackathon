import torch
import os

os.makedirs("model", exist_ok=True)

classifier = torch.nn.Sequential(
    torch.nn.Linear(768, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)

torch.save(classifier.state_dict(), "model/wav2vec_classifier.pt")

print("✅ Dummy classifier weights created successfully")
