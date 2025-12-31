import os
import tempfile
from typing import List, Dict, Any
import autochord
import librosa

def detect_key(chords: List[tuple]) -> str:
    """Detect the likely key based on chord frequency."""
    if not chords:
        return "C"
    
    chord_counts: Dict[str, int] = {}
    for chord, _, _ in chords:
        root = chord.split(":")[0] if ":" in chord else chord.split("m")[0].split("7")[0]
        root = root.replace("b", "").replace("#", "")[:1].upper()
        if root:
            chord_counts[root] = chord_counts.get(root, 0) + 1
    
    if chord_counts:
        return max(chord_counts, key=chord_counts.get)
    return "C"

def format_chord_name(chord: str) -> str:
    """Convert autochord format to simple chord names."""
    if not chord or chord == "N":
        return "N/C"
    
    # autochord returns chords like "C:maj", "A:min", "G:7"
    if ":" in chord:
        root, quality = chord.split(":", 1)
        if quality == "maj":
            return root
        elif quality == "min":
            return f"{root}m"
        elif quality == "7":
            return f"{root}7"
        elif quality == "maj7":
            return f"{root}maj7"
        elif quality == "min7":
            return f"{root}m7"
        else:
            return f"{root}{quality}"
    return chord

async def analyze_audio(file_path: str) -> Dict[str, Any]:
    """Analyze audio file and detect chords using autochord."""
    
    # Get audio duration
    y, sr = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Detect chords using autochord
    try:
        raw_chords = autochord.recognize(file_path)
    except Exception as e:
        print(f"autochord error: {e}")
        # Fallback to empty result
        raw_chords = []
    
    # Format results
    chords = []
    for chord_data in raw_chords:
        if len(chord_data) >= 3:
            chord_name, start_time, end_time = chord_data[0], chord_data[1], chord_data[2]
            formatted_chord = format_chord_name(chord_name)
            
            # Calculate confidence (autochord doesn't provide this, so we estimate)
            confidence = 0.85 if formatted_chord != "N/C" else 0.3
            
            chords.append({
                "chord": formatted_chord,
                "startTime": float(start_time),
                "endTime": float(end_time),
                "confidence": confidence,
                "notes": []
            })
    
    # Post-process: merge adjacent same chords
    merged_chords = []
    for chord in chords:
        if merged_chords and merged_chords[-1]["chord"] == chord["chord"]:
            merged_chords[-1]["endTime"] = chord["endTime"]
        else:
            merged_chords.append(chord)
    
    # Filter out very short N/C segments
    filtered_chords = [
        c for c in merged_chords 
        if c["chord"] != "N/C" or (c["endTime"] - c["startTime"]) > 1.0
    ]
    
    # Detect key
    raw_for_key = [(c["chord"], c["startTime"], c["endTime"]) for c in filtered_chords]
    detected_key = detect_key(raw_for_key)
    
    return {
        "chords": filtered_chords,
        "duration": duration,
        "key": detected_key
    }
