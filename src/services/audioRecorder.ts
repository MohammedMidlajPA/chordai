export interface RecordingResult {
  blob: Blob;
  url: string;
  duration: number;
}

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private startTime: number = 0;
  private stream: MediaStream | null = null;

  async startRecording(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        } 
      });
      
      this.audioChunks = [];
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: this.getSupportedMimeType(),
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.startTime = Date.now();
      this.mediaRecorder.start(100); // Collect data every 100ms
    } catch (error) {
      console.error('Error starting recording:', error);
      throw new Error('Could not access microphone. Please ensure microphone permissions are granted.');
    }
  }

  async stopRecording(): Promise<RecordingResult> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('No recording in progress'));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const duration = (Date.now() - this.startTime) / 1000;
        const blob = new Blob(this.audioChunks, { type: this.getSupportedMimeType() });
        const url = URL.createObjectURL(blob);
        
        // Stop all tracks
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
        }
        
        resolve({ blob, url, duration });
      };

      this.mediaRecorder.onerror = (event) => {
        reject(new Error('Recording error occurred'));
      };

      this.mediaRecorder.stop();
    });
  }

  isRecording(): boolean {
    return this.mediaRecorder?.state === 'recording';
  }

  private getSupportedMimeType(): string {
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];

    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        return mimeType;
      }
    }

    return 'audio/webm'; // Default fallback
  }

  // Clean up resources
  cleanup(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    this.mediaRecorder = null;
    this.audioChunks = [];
  }
}

// Singleton instance for easy use
export const audioRecorder = new AudioRecorder();
