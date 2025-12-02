import { useState, useEffect, useCallback } from 'react';
import { DetectedChord } from '@/services/chordDetector';

export interface Song {
  id: string;
  name: string;
  duration: number;
  chords: DetectedChord[];
  audioUrl: string; // blob URL
  audioData?: string; // base64 encoded for persistence (optional, for small files)
  type: 'upload' | 'recording';
  createdAt: string;
  accuracy: number;
}

interface SongLibraryState {
  songs: Song[];
  isLoading: boolean;
}

const STORAGE_KEY = 'chordai_song_library';
const MAX_AUDIO_SIZE_FOR_STORAGE = 2 * 1024 * 1024; // 2MB max for localStorage audio

export const useSongLibrary = () => {
  const [state, setState] = useState<SongLibraryState>({
    songs: [],
    isLoading: true,
  });

  // Load songs from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsedSongs: Song[] = JSON.parse(stored);
        
        // Recreate blob URLs from stored audio data
        const songsWithUrls = parsedSongs.map(song => {
          if (song.audioData) {
            // Convert base64 back to blob URL
            const byteCharacters = atob(song.audioData);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
              byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'audio/webm' });
            return { ...song, audioUrl: URL.createObjectURL(blob) };
          }
          return song;
        });
        
        setState({ songs: songsWithUrls, isLoading: false });
      } else {
        setState({ songs: [], isLoading: false });
      }
    } catch (error) {
      console.error('Error loading song library:', error);
      setState({ songs: [], isLoading: false });
    }
  }, []);

  // Save songs to localStorage whenever they change
  const saveSongs = useCallback((songs: Song[]) => {
    try {
      // Store only metadata and small audio files
      const songsToStore = songs.map(({ audioUrl, ...song }) => song);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(songsToStore));
    } catch (error) {
      console.error('Error saving song library:', error);
      // If storage is full, try removing audio data
      try {
        const songsWithoutAudio = songs.map(({ audioUrl, audioData, ...song }) => song);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(songsWithoutAudio));
      } catch (e) {
        console.error('Could not save to localStorage:', e);
      }
    }
  }, []);

  const addSong = useCallback(async (
    name: string,
    chords: DetectedChord[],
    audioBlob: Blob,
    type: 'upload' | 'recording',
    duration: number
  ): Promise<Song> => {
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Calculate average accuracy
    const accuracy = chords.length > 0
      ? Math.round(chords.reduce((sum, c) => sum + c.confidence, 0) / chords.length * 100)
      : 0;

    // Try to store audio data if small enough
    let audioData: string | undefined;
    if (audioBlob.size <= MAX_AUDIO_SIZE_FOR_STORAGE) {
      try {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        audioData = btoa(binary);
      } catch (error) {
        console.warn('Could not encode audio for storage:', error);
      }
    }

    const newSong: Song = {
      id: `song_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      duration,
      chords,
      audioUrl,
      audioData,
      type,
      createdAt: new Date().toISOString(),
      accuracy,
    };

    setState(prev => {
      const updatedSongs = [newSong, ...prev.songs];
      saveSongs(updatedSongs);
      return { ...prev, songs: updatedSongs };
    });

    return newSong;
  }, [saveSongs]);

  const removeSong = useCallback((songId: string) => {
    setState(prev => {
      const song = prev.songs.find(s => s.id === songId);
      if (song?.audioUrl) {
        URL.revokeObjectURL(song.audioUrl);
      }
      
      const updatedSongs = prev.songs.filter(s => s.id !== songId);
      saveSongs(updatedSongs);
      return { ...prev, songs: updatedSongs };
    });
  }, [saveSongs]);

  const updateSong = useCallback((songId: string, updates: Partial<Song>) => {
    setState(prev => {
      const updatedSongs = prev.songs.map(song =>
        song.id === songId ? { ...song, ...updates } : song
      );
      saveSongs(updatedSongs);
      return { ...prev, songs: updatedSongs };
    });
  }, [saveSongs]);

  const getSong = useCallback((songId: string): Song | undefined => {
    return state.songs.find(s => s.id === songId);
  }, [state.songs]);

  const clearLibrary = useCallback(() => {
    // Revoke all blob URLs
    state.songs.forEach(song => {
      if (song.audioUrl) {
        URL.revokeObjectURL(song.audioUrl);
      }
    });
    
    localStorage.removeItem(STORAGE_KEY);
    setState({ songs: [], isLoading: false });
  }, [state.songs]);

  // Stats
  const stats = {
    totalSongs: state.songs.length,
    uploadedSongs: state.songs.filter(s => s.type === 'upload').length,
    recordedSongs: state.songs.filter(s => s.type === 'recording').length,
    averageAccuracy: state.songs.length > 0
      ? Math.round(state.songs.reduce((sum, s) => sum + s.accuracy, 0) / state.songs.length)
      : 0,
    totalDuration: state.songs.reduce((sum, s) => sum + s.duration, 0),
  };

  return {
    songs: state.songs,
    isLoading: state.isLoading,
    stats,
    addSong,
    removeSong,
    updateSong,
    getSong,
    clearLibrary,
  };
};

export default useSongLibrary;
