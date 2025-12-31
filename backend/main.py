import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.chord_detector import analyze_audio

app = FastAPI(
    title="ChordAI API",
    description="AI-powered chord detection API using autochord",
    version="1.0.0"
)

# CORS configuration - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ChordAI API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/detect-chords")
async def detect_chords(file: UploadFile = File(...)):
    """
    Detect chords from an uploaded audio file.
    
    Accepts: MP3, WAV, FLAC, OGG, M4A audio files
    Returns: List of detected chords with timestamps
    """
    
    # Validate file type
    allowed_types = [
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", 
        "audio/flac", "audio/ogg", "audio/m4a", "audio/x-m4a",
        "audio/webm", "audio/x-wav"
    ]
    
    if file.content_type and file.content_type not in allowed_types:
        # Also check by extension
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Please upload MP3, WAV, FLAC, OGG, or M4A."
            )
    
    # Save uploaded file to temp location
    try:
        ext = os.path.splitext(file.filename or ".mp3")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Analyze audio
        result = await analyze_audio(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
