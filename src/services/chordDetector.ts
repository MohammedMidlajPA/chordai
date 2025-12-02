import { Chord, Note } from 'tonal';
import Pitchfinder from 'pitchfinder';

export interface DetectedChord {
  chord: string;
  startTime: number;
  endTime: number;
  confidence: number;
  notes: string[];
}

export interface ChordDetectionResult {
  chords: DetectedChord[];
  duration: number;
  tempo?: number;
}

// Note frequencies for reference (A4 = 440Hz)
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

class ChordDetector {
  private audioContext: AudioContext | null = null;
  private detectPitch: ReturnType<typeof Pitchfinder.YIN>;

  constructor() {
    this.detectPitch = Pitchfinder.YIN({ sampleRate: 44100 });
  }

  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: 44100 });
    }
    return this.audioContext;
  }

  private frequencyToNote(frequency: number): string | null {
    if (!frequency || frequency < 20 || frequency > 5000) return null;
    
    // Calculate MIDI note number
    const midiNote = Math.round(12 * Math.log2(frequency / 440) + 69);
    const noteIndex = midiNote % 12;
    const octave = Math.floor(midiNote / 12) - 1;
    
    return `${NOTE_NAMES[noteIndex]}${octave}`;
  }

  private noteNameOnly(noteWithOctave: string): string {
    return noteWithOctave.replace(/[0-9]/g, '');
  }

  private detectChordFromNotes(notes: string[]): string {
    if (notes.length === 0) return 'N/C';
    
    // Get unique note names without octaves
    const uniqueNotes = [...new Set(notes.map(n => this.noteNameOnly(n)))];
    
    if (uniqueNotes.length === 1) {
      return uniqueNotes[0]; // Single note
    }

    // Try to detect chord using tonal
    const detected = Chord.detect(uniqueNotes);
    if (detected.length > 0) {
      return detected[0];
    }

    // Fallback: return the root note
    return uniqueNotes[0];
  }

  async processAudioFile(file: File): Promise<ChordDetectionResult> {
    const audioContext = this.getAudioContext();
    
    // Read file as ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    return this.analyzeAudioBuffer(audioBuffer);
  }

  async processAudioBlob(blob: Blob): Promise<ChordDetectionResult> {
    const audioContext = this.getAudioContext();
    
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    return this.analyzeAudioBuffer(audioBuffer);
  }

  async processAudioUrl(url: string): Promise<ChordDetectionResult> {
    const audioContext = this.getAudioContext();
    
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    return this.analyzeAudioBuffer(audioBuffer);
  }

  private analyzeAudioBuffer(audioBuffer: AudioBuffer): ChordDetectionResult {
    const channelData = audioBuffer.getChannelData(0); // Get mono channel
    const sampleRate = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;

    // Analyze in chunks (every 0.5 seconds for chord detection)
    const chunkDuration = 0.5; // seconds
    const chunkSize = Math.floor(sampleRate * chunkDuration);
    const chords: DetectedChord[] = [];

    let currentChord: DetectedChord | null = null;

    for (let i = 0; i < channelData.length; i += chunkSize) {
      const chunk = channelData.slice(i, i + chunkSize);
      const startTime = i / sampleRate;
      const endTime = Math.min((i + chunkSize) / sampleRate, duration);

      // Detect pitches in this chunk using multiple windows
      const detectedNotes = this.detectNotesInChunk(chunk, sampleRate);
      const chordName = this.detectChordFromNotes(detectedNotes);

      // If same chord continues, extend it
      if (currentChord && currentChord.chord === chordName) {
        currentChord.endTime = endTime;
      } else {
        // Save previous chord if exists
        if (currentChord) {
          chords.push(currentChord);
        }
        
        // Start new chord
        currentChord = {
          chord: chordName,
          startTime,
          endTime,
          confidence: this.calculateConfidence(detectedNotes),
          notes: detectedNotes,
        };
      }
    }

    // Add last chord
    if (currentChord) {
      chords.push(currentChord);
    }

    // Filter out very short detections and N/C chords at edges
    const filteredChords = this.postProcessChords(chords);

    return {
      chords: filteredChords,
      duration,
    };
  }

  private detectNotesInChunk(chunk: Float32Array, sampleRate: number): string[] {
    const notes: string[] = [];
    const windowSize = 2048;
    const hopSize = 512;

    // Use overlapping windows for better detection
    for (let j = 0; j < chunk.length - windowSize; j += hopSize) {
      const window = chunk.slice(j, j + windowSize);
      
      // Apply Hanning window
      const windowed = this.applyHanningWindow(window);
      
      // Detect pitch
      const frequency = this.detectPitch(windowed);
      
      if (frequency) {
        const note = this.frequencyToNote(frequency);
        if (note) {
          notes.push(note);
        }
      }
    }

    // Also use FFT-based harmonic detection for chord notes
    const harmonicNotes = this.detectHarmonicsFFT(chunk, sampleRate);
    notes.push(...harmonicNotes);

    return notes;
  }

  private applyHanningWindow(data: Float32Array): Float32Array {
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const multiplier = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (data.length - 1)));
      result[i] = data[i] * multiplier;
    }
    return result;
  }

  private detectHarmonicsFFT(chunk: Float32Array, sampleRate: number): string[] {
    const notes: string[] = [];
    const fftSize = 4096;
    
    // Simple FFT-based peak detection
    if (chunk.length < fftSize) return notes;

    // Calculate energy in frequency bands for each note
    const chromagram = new Array(12).fill(0);
    
    // Simple DFT for specific frequencies (simplified approach)
    for (let noteIdx = 0; noteIdx < 12; noteIdx++) {
      // Check multiple octaves
      for (let octave = 2; octave <= 6; octave++) {
        const freq = 440 * Math.pow(2, (noteIdx - 9 + (octave - 4) * 12) / 12);
        const energy = this.goertzel(chunk, freq, sampleRate);
        chromagram[noteIdx] += energy;
      }
    }

    // Find peaks in chromagram
    const maxEnergy = Math.max(...chromagram);
    const threshold = maxEnergy * 0.3;

    for (let i = 0; i < 12; i++) {
      if (chromagram[i] > threshold && chromagram[i] > 0.01) {
        notes.push(NOTE_NAMES[i] + '4'); // Default octave
      }
    }

    return notes;
  }

  private goertzel(samples: Float32Array, targetFreq: number, sampleRate: number): number {
    const k = Math.round((samples.length * targetFreq) / sampleRate);
    const w = (2 * Math.PI * k) / samples.length;
    const cosine = Math.cos(w);
    const coeff = 2 * cosine;

    let s0 = 0, s1 = 0, s2 = 0;

    for (let i = 0; i < samples.length; i++) {
      s0 = samples[i] + coeff * s1 - s2;
      s2 = s1;
      s1 = s0;
    }

    return Math.sqrt(s1 * s1 + s2 * s2 - coeff * s1 * s2) / samples.length;
  }

  private calculateConfidence(notes: string[]): number {
    if (notes.length === 0) return 0;
    
    // More notes detected = higher confidence
    const uniqueNotes = new Set(notes.map(n => this.noteNameOnly(n)));
    
    if (uniqueNotes.size >= 3) return 0.85;
    if (uniqueNotes.size >= 2) return 0.7;
    return 0.5;
  }

  private postProcessChords(chords: DetectedChord[]): DetectedChord[] {
    // Filter out very short chords (< 0.3 seconds)
    let filtered = chords.filter(c => (c.endTime - c.startTime) >= 0.3);
    
    // Merge adjacent same chords
    const merged: DetectedChord[] = [];
    for (const chord of filtered) {
      const last = merged[merged.length - 1];
      if (last && last.chord === chord.chord && (chord.startTime - last.endTime) < 0.2) {
        last.endTime = chord.endTime;
        last.confidence = Math.max(last.confidence, chord.confidence);
      } else {
        merged.push({ ...chord });
      }
    }

    // Replace N/C with previous chord if surrounded
    for (let i = 1; i < merged.length - 1; i++) {
      if (merged[i].chord === 'N/C') {
        if (merged[i - 1].chord === merged[i + 1].chord) {
          merged[i].chord = merged[i - 1].chord;
        }
      }
    }

    return merged.filter(c => c.chord !== 'N/C' || (c.endTime - c.startTime) > 1);
  }

  // Get chord for guitar display
  getGuitarChord(chordName: string): { name: string; positions: number[]; fingers: number[] } | null {
    // Common guitar chord fingerings
    const guitarChords: Record<string, { positions: number[]; fingers: number[] }> = {
      'C': { positions: [-1, 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0] },
      'Cm': { positions: [-1, 3, 5, 5, 4, 3], fingers: [0, 1, 3, 4, 2, 1] },
      'D': { positions: [-1, -1, 0, 2, 3, 2], fingers: [0, 0, 0, 1, 3, 2] },
      'Dm': { positions: [-1, -1, 0, 2, 3, 1], fingers: [0, 0, 0, 2, 3, 1] },
      'E': { positions: [0, 2, 2, 1, 0, 0], fingers: [0, 2, 3, 1, 0, 0] },
      'Em': { positions: [0, 2, 2, 0, 0, 0], fingers: [0, 2, 3, 0, 0, 0] },
      'F': { positions: [1, 3, 3, 2, 1, 1], fingers: [1, 3, 4, 2, 1, 1] },
      'Fm': { positions: [1, 3, 3, 1, 1, 1], fingers: [1, 3, 4, 1, 1, 1] },
      'G': { positions: [3, 2, 0, 0, 0, 3], fingers: [2, 1, 0, 0, 0, 3] },
      'Gm': { positions: [3, 5, 5, 3, 3, 3], fingers: [1, 3, 4, 1, 1, 1] },
      'A': { positions: [-1, 0, 2, 2, 2, 0], fingers: [0, 0, 1, 2, 3, 0] },
      'Am': { positions: [-1, 0, 2, 2, 1, 0], fingers: [0, 0, 2, 3, 1, 0] },
      'B': { positions: [-1, 2, 4, 4, 4, 2], fingers: [0, 1, 2, 3, 4, 1] },
      'Bm': { positions: [-1, 2, 4, 4, 3, 2], fingers: [0, 1, 3, 4, 2, 1] },
      'C7': { positions: [-1, 3, 2, 3, 1, 0], fingers: [0, 3, 2, 4, 1, 0] },
      'D7': { positions: [-1, -1, 0, 2, 1, 2], fingers: [0, 0, 0, 2, 1, 3] },
      'E7': { positions: [0, 2, 0, 1, 0, 0], fingers: [0, 2, 0, 1, 0, 0] },
      'G7': { positions: [3, 2, 0, 0, 0, 1], fingers: [3, 2, 0, 0, 0, 1] },
      'A7': { positions: [-1, 0, 2, 0, 2, 0], fingers: [0, 0, 1, 0, 2, 0] },
      'Am7': { positions: [-1, 0, 2, 0, 1, 0], fingers: [0, 0, 2, 0, 1, 0] },
      'Cmaj7': { positions: [-1, 3, 2, 0, 0, 0], fingers: [0, 3, 2, 0, 0, 0] },
      'Dmaj7': { positions: [-1, -1, 0, 2, 2, 2], fingers: [0, 0, 0, 1, 1, 1] },
      'Fmaj7': { positions: [-1, -1, 3, 2, 1, 0], fingers: [0, 0, 3, 2, 1, 0] },
      'Gmaj7': { positions: [3, 2, 0, 0, 0, 2], fingers: [2, 1, 0, 0, 0, 3] },
    };

    // Normalize chord name
    const normalizedName = chordName.replace('M', '').replace('maj', 'maj7').trim();
    
    // Try exact match first
    if (guitarChords[normalizedName]) {
      return { name: chordName, ...guitarChords[normalizedName] };
    }

    // Try simplified match (just root + quality)
    const root = chordName.match(/^[A-G][#b]?/)?.[0];
    if (root) {
      const isMinor = chordName.toLowerCase().includes('m') && !chordName.toLowerCase().includes('maj');
      const simpleChord = isMinor ? `${root}m` : root;
      if (guitarChords[simpleChord]) {
        return { name: chordName, ...guitarChords[simpleChord] };
      }
    }

    return null;
  }

  cleanup(): void {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

export const chordDetector = new ChordDetector();
