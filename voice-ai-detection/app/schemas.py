from pydantic import BaseModel, Field

class VoiceRequest(BaseModel):
    language: str = Field(..., example="Tamil")
    audioFormat: str = Field(..., example="mp3")
    audioBase64: str

class VoiceResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str
