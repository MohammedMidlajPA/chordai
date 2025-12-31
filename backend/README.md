# ChordAI Backend

Python FastAPI backend with **BTC (Bi-Directional Transformer for Chord Recognition)** deep learning model for accurate chord detection from audio files.

## Features

- **~85% accuracy** chord detection using pre-trained BTC Transformer model
- Supports MP3, WAV, and other audio formats
- Returns chord names compatible with guitar chord diagrams
- Real-time API for audio analysis

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Health Check
```
GET /health
```

### Detect Chords
```
POST /api/detect-chords
Content-Type: multipart/form-data
Body: file=<audio file>
```

**Response:**
```json
{
  "chords": [
    {
      "chord": "C",
      "startTime": 0.0,
      "endTime": 2.5,
      "confidence": 0.85,
      "notes": []
    },
    {
      "chord": "Am",
      "startTime": 2.5,
      "endTime": 5.0,
      "confidence": 0.85,
      "notes": []
    }
  ],
  "duration": 180.5,
  "key": "C"
}
```

## Model

This backend uses the **BTC-ISMIR19** model:
- Paper: "A Bi-Directional Transformer for Musical Chord Recognition"
- Pre-trained on large chord dataset
- Recognizes 24 chord types (major/minor for all 12 notes)

## Chord Format

The API returns guitar-friendly chord names:
- Major chords: C, D, E, F, G, A, B
- Minor chords: Cm, Dm, Em, Fm, Gm, Am, Bm
- Sharp notes: C#, F#, G# (or flats: Bb, Eb, Ab)

## Requirements

- Python 3.10+
- PyTorch (CPU or GPU)
- librosa for audio processing
