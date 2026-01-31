import torch
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ----------------------------
# Globals (lazy-loaded)
# ----------------------------
processor = None
wav2vec = None
classifier = None
MODEL_LOADED = False

MODEL_PATH = "model/wav2vec_classifier.pt"


def load_model():
    """
    Loads the Wav2Vec2 backbone and classifier weights safely.
    This function is called ONLY on the first request.
    """
    global processor, wav2vec, classifier, MODEL_LOADED

    if MODEL_LOADED:
        return

    print("🔄 Loading Wav2Vec2 model (first request only)...")

    # Load Wav2Vec2 components
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    wav2vec = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )

    # Define classifier head
    classifier = torch.nn.Sequential(
        torch.nn.Linear(768, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid()
    )

    # Load classifier weights safely
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu"
            )
            classifier.load_state_dict(state_dict)
            print("✅ Classifier weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load classifier weights: {e}")
            print("⚠️ Using randomly initialized classifier")
    else:
        print("⚠️ Classifier weights file not found, using random weights")

    wav2vec.eval()
    classifier.eval()

    MODEL_LOADED = True
    print("✅ Model is ready for inference")


def predict(signal):
    """
    Runs inference on a single audio signal.
    Returns a probability between 0.0 and 1.0
    """
    if not MODEL_LOADED:
        load_model()

    # Prepare input
    inputs = processor(
        signal,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        # Extract embeddings
        embeddings = wav2vec(**inputs).last_hidden_state

        # Mean pooling
        pooled = embeddings.mean(dim=1)

        # Classification
        prob = classifier(pooled).item()

    return prob
