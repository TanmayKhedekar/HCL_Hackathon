from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_api_key
from app.schemas import VoiceRequest, VoiceResponse
from app.audio_utils import decode_and_preprocess
from app.model import predict
from app.explanation import generate_explanation

# ----------------------------
# Supported Languages
# ----------------------------
SUPPORTED_LANGS = {
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
}

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human",
    version="1.0.0"
)

# ----------------------------
# CORS (Required for Swagger & browser testing)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Hackathon-safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# API Endpoint
# ----------------------------
@app.post(
    "/api/voice-detection",
    response_model=VoiceResponse
)
def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    # ----------------------------
    # Validate language
    # ----------------------------
    if request.language not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language"
        )

    # ----------------------------
    # Validate audio format
    # ----------------------------
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Invalid audio format. Only mp3 is supported"
        )

    # ----------------------------
    # Decode & preprocess audio
    # ----------------------------
    try:
        signal = decode_and_preprocess(request.audioBase64)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted MP3 audio"
        )

    # ----------------------------
    # Run ML inference
    # ----------------------------
    try:
        prob = predict(signal)
    except Exception as e:
        # Absolute safety net (should not happen now)
        raise HTTPException(
            status_code=500,
            detail="Voice analysis failed"
        )

    # ----------------------------
    # Classification
    # ----------------------------
    classification = (
        "AI_GENERATED" if prob >= 0.5 else "HUMAN"
    )

    # ----------------------------
    # Success Response
    # ----------------------------
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(float(prob), 2),
        "explanation": generate_explanation(prob)
    }
