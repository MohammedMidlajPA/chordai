# ChordAI Backend

Python FastAPI backend for chord detection using autochord ML library.

## Setup

### Prerequisites
- Python 3.9+
- FFmpeg (required for audio processing)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at http://localhost:8000

## API Endpoints

### GET /
Health check, returns API version.

### GET /health
Returns health status.

### POST /api/detect-chords
Upload an audio file to detect chords.

**Request:**
- Content-Type: multipart/form-data
- Body: file (audio file: MP3, WAV, FLAC, OGG, M4A)

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
      "chord": "G",
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

## Deployment

### Railway
```bash
railway init
railway up
```

### Render
Create a new Web Service and connect your GitHub repo.

### Docker
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
