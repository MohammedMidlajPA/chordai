import { Chord } from 'tonal';
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
  key?: string;
}

// API URL - change this when deployed
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Note frequencies for reference (A4 = 440Hz)
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

class ChordDetector {
  private audioContext: AudioContext | null = null;
  private detectPitch: ReturnType<typeof Pitchfinder.YIN>;
  private useBackend: boolean = true; // Try backend first

  constructor() {
    this.detectPitch = Pitchfinder.YIN({ sampleRate: 44100 });
  }

  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: 44100 });
    }
    return this.audioContext;
  }

  /**
   * Process audio file - tries Python backend first, falls back to JS
   */
  async processAudioFile(file: File): Promise<ChordDetectionResult> {
    // Try Python backend first for better accuracy
    if (this.useBackend) {
      try {
        const result = await this.callBackendAPI(file);
        console.log('Using Python ML backend for chord detection');
        return result;
      } catch (error) {
        console.warn('Backend unavailable, falling back to JS detection:', error);
        this.useBackend = false; // Don't try again this session
      }
    }

    // Fallback to JavaScript-based detection
    console.log('Using JavaScript frontend detection');
    return this.processAudioFileJS(file);
  }

  /**
   * Process audio blob - tries Python backend first, falls back to JS
   */
  async processAudioBlob(blob: Blob): Promise<ChordDetectionResult> {
    // Convert blob to file for API
    const file = new File([blob], 'recording.webm', { type: blob.type });
    
    if (this.useBackend) {
      try {
        const result = await this.callBackendAPI(file);
        console.log('Using Python ML backend for chord detection');
        return result;
      } catch (error) {
        console.warn('Backend unavailable, falling back to JS detection:', error);
        this.useBackend = false;
      }
    }

    return this.processAudioBlobJS(blob);
  }

  /**
   * Call the Python FastAPI backend
   */
  private async callBackendAPI(file: File): Promise<ChordDetectionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/api/detect-chords`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    
    // Transform API response to match our interface
    return {
      chords: data.chords.map((c: any) => ({
        chord: c.chord,
        startTime: c.startTime,
        endTime: c.endTime,
        confidence: c.confidence || 0.85,
        notes: c.notes || [],
      })),
      duration: data.duration,
      key: data.key,
    };
  }

  // ============== JavaScript Fallback Methods ==============

  private async processAudioFileJS(file: File): Promise<ChordDetectionResult> {
    const audioContext = this.getAudioContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return this.analyzeAudioBuffer(audioBuffer);
  }

  private async processAudioBlobJS(blob: Blob): Promise<ChordDetectionResult> {
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

  private frequencyToNote(frequency: number): string | null {
    if (!frequency || frequency < 20 || frequency > 5000) return null;
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
    const uniqueNotes = [...new Set(notes.map(n => this.noteNameOnly(n)))];
    if (uniqueNotes.length === 1) return uniqueNotes[0];
    const detected = Chord.detect(uniqueNotes);
    if (detected.length > 0) return detected[0];
    return uniqueNotes[0];
  }

  private analyzeAudioBuffer(audioBuffer: AudioBuffer): ChordDetectionResult {
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;
    const chunkDuration = 0.5;
    const chunkSize = Math.floor(sampleRate * chunkDuration);
    const chords: DetectedChord[] = [];
    let currentChord: DetectedChord | null = null;

    for (let i = 0; i < channelData.length; i += chunkSize) {
      const chunk = channelData.slice(i, i + chunkSize);
      const startTime = i / sampleRate;
      const endTime = Math.min((i + chunkSize) / sampleRate, duration);
      const detectedNotes = this.detectNotesInChunk(chunk, sampleRate);
      const chordName = this.detectChordFromNotes(detectedNotes);

      if (currentChord && currentChord.chord === chordName) {
        currentChord.endTime = endTime;
      } else {
        if (currentChord) chords.push(currentChord);
        currentChord = {
          chord: chordName,
          startTime,
          endTime,
          confidence: this.calculateConfidence(detectedNotes),
          notes: detectedNotes,
        };
      }
    }

    if (currentChord) chords.push(currentChord);
    return { chords: this.postProcessChords(chords), duration };
  }

  private detectNotesInChunk(chunk: Float32Array, sampleRate: number): string[] {
    const notes: string[] = [];
    const windowSize = 2048;
    const hopSize = 512;

    for (let j = 0; j < chunk.length - windowSize; j += hopSize) {
      const window = chunk.slice(j, j + windowSize);
      const windowed = this.applyHanningWindow(window);
      const frequency = this.detectPitch(windowed);
      if (frequency) {
        const note = this.frequencyToNote(frequency);
        if (note) notes.push(note);
      }
    }

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
    if (chunk.length < fftSize) return notes;

    const chromagram = new Array(12).fill(0);
    for (let noteIdx = 0; noteIdx < 12; noteIdx++) {
      for (let octave = 2; octave <= 6; octave++) {
        const freq = 440 * Math.pow(2, (noteIdx - 9 + (octave - 4) * 12) / 12);
        const energy = this.goertzel(chunk, freq, sampleRate);
        chromagram[noteIdx] += energy;
      }
    }

    const maxEnergy = Math.max(...chromagram);
    const threshold = maxEnergy * 0.3;

    for (let i = 0; i < 12; i++) {
      if (chromagram[i] > threshold && chromagram[i] > 0.01) {
        notes.push(NOTE_NAMES[i] + '4');
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
    const uniqueNotes = new Set(notes.map(n => this.noteNameOnly(n)));
    if (uniqueNotes.size >= 3) return 0.85;
    if (uniqueNotes.size >= 2) return 0.7;
    return 0.5;
  }

  private postProcessChords(chords: DetectedChord[]): DetectedChord[] {
    let filtered = chords.filter(c => (c.endTime - c.startTime) >= 0.3);
    
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

    for (let i = 1; i < merged.length - 1; i++) {
      if (merged[i].chord === 'N/C') {
        if (merged[i - 1].chord === merged[i + 1].chord) {
          merged[i].chord = merged[i - 1].chord;
        }
      }
    }

    return merged.filter(c => c.chord !== 'N/C' || (c.endTime - c.startTime) > 1);
  }

  getGuitarChord(chordName: string): { name: string; positions: number[]; fingers: number[] } | null {
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
    };

    const normalizedName = chordName.replace('M', '').trim();
    if (guitarChords[normalizedName]) {
      return { name: chordName, ...guitarChords[normalizedName] };
    }

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
