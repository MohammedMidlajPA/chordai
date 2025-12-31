import os
import tempfile
from typing import List, Dict, Any, Tuple
import numpy as np
import librosa

# Chord templates - 12 pitch classes for each chord type
CHORD_TEMPLATES = {
    # Major chords (root, major 3rd, perfect 5th)
    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'G#': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    
    # Minor chords (root, minor 3rd, perfect 5th)
    'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'Bm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    
    # Dominant 7th chords
    'C7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'D7': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    'E7': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    'G7': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    'A7': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
}

# Convert templates to numpy arrays
for chord in CHORD_TEMPLATES:
    CHORD_TEMPLATES[chord] = np.array(CHORD_TEMPLATES[chord], dtype=float)
    # Normalize
    CHORD_TEMPLATES[chord] /= np.linalg.norm(CHORD_TEMPLATES[chord])


def match_chord(chroma: np.ndarray) -> Tuple[str, float]:
    """Match a chroma vector to the best chord template."""
    if np.sum(chroma) < 0.01:
        return 'N/C', 0.0
    
    # Normalize chroma
    chroma_norm = chroma / (np.linalg.norm(chroma) + 1e-6)
    
    best_chord = 'N/C'
    best_score = 0.0
    
    for chord_name, template in CHORD_TEMPLATES.items():
        # Cosine similarity
        score = np.dot(chroma_norm, template)
        if score > best_score:
            best_score = score
            best_chord = chord_name
    
    # Threshold - if score is too low, return N/C
    if best_score < 0.5:
        return 'N/C', best_score
    
    return best_chord, float(best_score)


def detect_key(chords: List[Dict]) -> str:
    """Detect the likely key based on chord frequency."""
    if not chords:
        return "C"
    
    chord_counts: Dict[str, int] = {}
    for c in chords:
        chord = c.get('chord', '')
        if chord == 'N/C':
            continue
        # Extract root note
        root = chord[0]
        if len(chord) > 1 and chord[1] == '#':
            root = chord[:2]
        chord_counts[root] = chord_counts.get(root, 0) + 1
    
    if chord_counts:
        return max(chord_counts, key=chord_counts.get)
    return "C"


async def analyze_audio(file_path: str) -> Dict[str, Any]:
    """Analyze audio file and detect chords using chromagram analysis."""
    
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Parameters for analysis
    hop_length = 512
    frame_length = 2048
    
    # Extract chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Segment into chunks (about 0.5 seconds each)
    chunk_frames = int(0.5 * sr / hop_length)
    
    chords = []
    n_frames = chroma.shape[1]
    
    current_chord = None
    current_start = 0.0
    
    for i in range(0, n_frames, chunk_frames):
        # Get average chroma for this chunk
        end_idx = min(i + chunk_frames, n_frames)
        chunk_chroma = np.mean(chroma[:, i:end_idx], axis=1)
        
        # Get time
        start_time = i * hop_length / sr
        end_time = end_idx * hop_length / sr
        
        # Match chord
        chord_name, confidence = match_chord(chunk_chroma)
        
        # Merge with previous if same chord
        if current_chord is not None and current_chord == chord_name:
            # Extend current chord
            pass
        else:
            # Save previous chord
            if current_chord is not None:
                chords.append({
                    'chord': current_chord,
                    'startTime': current_start,
                    'endTime': start_time,
                    'confidence': 0.75,
                    'notes': []
                })
            current_chord = chord_name
            current_start = start_time
    
    # Add last chord
    if current_chord is not None:
        chords.append({
            'chord': current_chord,
            'startTime': current_start,
            'endTime': duration,
            'confidence': 0.75,
            'notes': []
        })
    
    # Post-process: filter short N/C segments and merge
    filtered_chords = []
    for c in chords:
        dur = c['endTime'] - c['startTime']
        # Skip very short N/C
        if c['chord'] == 'N/C' and dur < 1.0:
            continue
        # Merge with previous if same
        if filtered_chords and filtered_chords[-1]['chord'] == c['chord']:
            filtered_chords[-1]['endTime'] = c['endTime']
        else:
            filtered_chords.append(c)
    
    # Detect key
    detected_key = detect_key(filtered_chords)
    
    return {
        'chords': filtered_chords,
        'duration': duration,
        'key': detected_key
    }
