import os
import sys
from typing import List, Dict, Any
import numpy as np
import librosa
import torch
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_model import BTC_model
from utils.hparams import HParams

# Chord index to name mapping
IDX2CHORD = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 
             'F', 'F:min', 'F#', 'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 
             'A#', 'A#:min', 'B', 'B:min', 'N']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
_model = None
_mean = None
_std = None
_config = None


def format_chord_for_guitar(chord: str) -> str:
    """Convert BTC chord notation to guitar-friendly format."""
    if chord == 'N' or chord == 'X':
        return 'N/C'
    
    # Convert notation like "C:min" to "Cm", "C#:min" to "C#m"
    chord = chord.replace(':min', 'm')
    chord = chord.replace(':maj', '')
    chord = chord.replace(':7', '7')
    chord = chord.replace(':dim', 'dim')
    chord = chord.replace(':aug', 'aug')
    chord = chord.replace(':sus4', 'sus4')
    chord = chord.replace(':sus2', 'sus2')
    
    # Handle flats - convert A# to Bb for common guitar usage
    replacements = {
        'A#': 'Bb', 'A#m': 'Bbm',
        'D#': 'Eb', 'D#m': 'Ebm', 
        'G#': 'Ab', 'G#m': 'Abm',
    }
    
    for old, new in replacements.items():
        if chord == old:
            return new
    
    return chord


def load_model():
    """Load the BTC model and weights."""
    global _model, _mean, _std, _config
    
    if _model is not None:
        return
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'run_config.yaml')
    _config = HParams.load(config_path)
    
    # Load model
    _model = BTC_model(config=_config.model).to(device)
    
    # Load weights
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'btc_model.pt')
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        _mean = checkpoint['mean']
        _std = checkpoint['std']
        _model.load_state_dict(checkpoint['model'])
        _model.eval()
        print(f"BTC model loaded successfully from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")


def extract_features(audio_path: str) -> tuple:
    """Extract CQT features from audio file."""
    global _config
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=_config.mp3['song_hz'], mono=True)
    duration = len(y) / sr
    
    # Extract CQT features in chunks
    current_pos = 0
    chunk_size = int(_config.mp3['song_hz'] * _config.mp3['inst_len'])
    feature = None
    
    while current_pos + chunk_size < len(y):
        chunk = y[current_pos:current_pos + chunk_size]
        cqt = librosa.cqt(chunk, sr=sr, 
                         n_bins=_config.feature['n_bins'],
                         bins_per_octave=_config.feature['bins_per_octave'],
                         hop_length=_config.feature['hop_length'])
        if feature is None:
            feature = cqt
        else:
            feature = np.concatenate((feature, cqt), axis=1)
        current_pos += chunk_size
    
    # Process remaining audio
    if current_pos < len(y):
        remaining = y[current_pos:]
        if len(remaining) > _config.feature['hop_length']:
            cqt = librosa.cqt(remaining, sr=sr,
                             n_bins=_config.feature['n_bins'],
                             bins_per_octave=_config.feature['bins_per_octave'],
                             hop_length=_config.feature['hop_length'])
            if feature is None:
                feature = cqt
            else:
                feature = np.concatenate((feature, cqt), axis=1)
    
    # Log magnitude
    feature = np.log(np.abs(feature) + 1e-6)
    
    # Calculate time per feature frame
    feature_per_second = _config.mp3['inst_len'] / _config.model['timestep']
    
    return feature, feature_per_second, duration


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
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
        chord_counts[root] = chord_counts.get(root, 0) + 1
    
    if chord_counts:
        return max(chord_counts, key=chord_counts.get)
    return "C"


async def analyze_audio(file_path: str) -> Dict[str, Any]:
    """Analyze audio file and detect chords using BTC Transformer model."""
    global _model, _mean, _std, _config
    
    # Load model if not already loaded
    load_model()
    
    # Extract features
    feature, time_unit, duration = extract_features(file_path)
    
    # Transpose and normalize
    feature = feature.T
    feature = (feature - _mean) / _std
    
    # Pad to timestep boundary
    n_timestep = _config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    if num_pad < n_timestep:
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    
    num_instances = feature.shape[0] // n_timestep
    
    # Run inference
    raw_chords = []
    start_time = 0.0
    prev_chord = None
    
    with torch.no_grad():
        _model.eval()
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        
        for t in range(num_instances):
            chunk = feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
            encoder_output, _ = _model.self_attn_layers(chunk)
            prediction, _ = _model.output_layer(encoder_output)
            prediction = prediction.squeeze()
            
            for i in range(n_timestep):
                current_time = time_unit * (n_timestep * t + i)
                chord_idx = prediction[i].item()
                
                if prev_chord is None:
                    prev_chord = chord_idx
                    start_time = current_time
                    continue
                
                if chord_idx != prev_chord:
                    # Save previous chord
                    chord_name = IDX2CHORD[prev_chord] if prev_chord < len(IDX2CHORD) else 'N'
                    raw_chords.append({
                        'start': start_time,
                        'end': current_time,
                        'chord': chord_name
                    })
                    start_time = current_time
                    prev_chord = chord_idx
                
                # Handle last segment
                if t == num_instances - 1 and i + num_pad >= n_timestep:
                    chord_name = IDX2CHORD[prev_chord] if prev_chord < len(IDX2CHORD) else 'N'
                    raw_chords.append({
                        'start': start_time,
                        'end': duration,
                        'chord': chord_name
                    })
                    break
    
    # Post-process chords
    chords = []
    for c in raw_chords:
        chord_name = format_chord_for_guitar(c['chord'])
        
        # Skip very short N/C chords
        chord_duration = c['end'] - c['start']
        if chord_name == 'N/C' and chord_duration < 0.5:
            continue
        
        # Merge with previous if same chord
        if chords and chords[-1]['chord'] == chord_name:
            chords[-1]['endTime'] = c['end']
        else:
            chords.append({
                'chord': chord_name,
                'startTime': c['start'],
                'endTime': c['end'],
                'confidence': 0.85,
                'notes': []
            })
    
    # Detect key
    detected_key = detect_key(chords)
    
    return {
        'chords': chords,
        'duration': duration,
        'key': detected_key
    }
