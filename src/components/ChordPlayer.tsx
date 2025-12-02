import { useState, useRef, useEffect, useMemo } from 'react';
import { Play, Pause, SkipBack, Volume2 } from 'lucide-react';
import { DetectedChord } from '@/services/chordDetector';
import { GuitarChord } from './GuitarChord';

interface ChordPlayerProps {
  audioUrl: string;
  chords: DetectedChord[];
  songName: string;
}

export const ChordPlayer = ({ audioUrl, chords, songName }: ChordPlayerProps) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Get unique chords
  const uniqueChords = useMemo(() => {
    const seen = new Set<string>();
    return chords.filter(c => {
      if (seen.has(c.chord) || c.chord === 'N/C') return false;
      seen.add(c.chord);
      return true;
    });
  }, [chords]);

  // Detect key
  const detectedKey = useMemo(() => {
    const counts: Record<string, number> = {};
    chords.forEach(c => {
      if (c.chord !== 'N/C') {
        const root = c.chord.match(/^[A-G][#b]?/)?.[0] || c.chord;
        counts[root] = (counts[root] || 0) + 1;
      }
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'C';
  }, [chords]);

  // Find current chord
  const currentChordIndex = useMemo(() => {
    return chords.findIndex(c => currentTime >= c.startTime && currentTime < c.endTime);
  }, [chords, currentTime]);

  const currentChord = currentChordIndex >= 0 ? chords[currentChordIndex] : null;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setDuration(audio.duration);
    const onEnded = () => { setIsPlaying(false); setCurrentTime(0); };

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('ended', onEnded);

    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('ended', onEnded);
    };
  }, []);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.load();
    }
  }, [audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    isPlaying ? audioRef.current.pause() : audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const seekTo = (time: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = time;
    setCurrentTime(time);
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    seekTo(percent * duration);
  };

  const formatTime = (t: number) => {
    if (isNaN(t)) return '00:00';
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-[#f5f5f5] rounded-xl overflow-hidden">
      <audio ref={audioRef} preload="metadata" />
      
      {/* Header with song name */}
      <div className="bg-white px-4 py-3 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 truncate">{songName}</h2>
      </div>

      {/* Player Controls */}
      <div className="bg-white px-4 py-3 flex items-center gap-3 border-b border-gray-200">
        <button
          onClick={() => seekTo(0)}
          className="p-2 text-gray-500 hover:text-gray-700"
        >
          <SkipBack className="h-5 w-5" />
        </button>
        
        <button
          onClick={togglePlay}
          className="w-10 h-10 rounded-full bg-[#00c7b7] hover:bg-[#00b5a6] flex items-center justify-center text-white"
        >
          {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 ml-0.5" />}
        </button>

        <span className="text-sm text-gray-600 min-w-[45px]">{formatTime(currentTime)}</span>

        {/* Progress Bar */}
        <div 
          className="flex-1 h-1 bg-gray-200 rounded-full cursor-pointer"
          onClick={handleProgressClick}
        >
          <div 
            className="h-full bg-[#00c7b7] rounded-full"
            style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>

        <span className="text-sm text-gray-600 min-w-[45px]">{formatTime(duration)}</span>
        
        <button className="p-2 text-gray-500 hover:text-gray-700">
          <Volume2 className="h-5 w-5" />
        </button>
      </div>

      {/* Chord Timeline */}
      <div className="flex h-12 bg-[#e8e8e8] overflow-x-auto scrollbar-hide">
        {chords.map((chord, i) => {
          const isActive = i === currentChordIndex;
          const w = duration ? Math.max(((chord.endTime - chord.startTime) / duration) * 100, 4) : 8;
          
          return (
            <button
              key={i}
              onClick={() => seekTo(chord.startTime)}
              className={`h-full flex items-center justify-center border-r border-gray-300 shrink-0 transition-colors ${
                isActive ? 'bg-[#d0d0d0]' : 'bg-[#e8e8e8] hover:bg-[#ddd]'
              }`}
              style={{ width: `${w}%`, minWidth: '40px' }}
            >
              <span className={`font-bold text-sm ${isActive ? 'text-gray-900' : 'text-gray-500'}`}>
                {chord.chord}
              </span>
            </button>
          );
        })}
      </div>

      {/* Chord Diagrams Section */}
      <div className="bg-white p-4">
        <div className="flex items-center gap-4 mb-3">
          <span className="text-sm font-medium text-gray-700 flex items-center gap-1">
            <span className="text-[#00c7b7]">✓</span> Chord diagrams
          </span>
          <span className="text-sm text-gray-400">Chord overview</span>
        </div>
        
        <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
          {uniqueChords.slice(0, 8).map((chord, i) => (
            <div 
              key={i}
              onClick={() => {
                const first = chords.find(c => c.chord === chord.chord);
                if (first) seekTo(first.startTime);
              }}
              className="shrink-0 cursor-pointer"
            >
              <GuitarChord 
                chord={chord.chord} 
                size="md"
                isActive={currentChord?.chord === chord.chord}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Song Details Footer */}
      <div className="bg-gray-50 px-4 py-3 border-t border-gray-200 flex items-center justify-between text-sm">
        <div className="flex items-center gap-6">
          <div>
            <span className="text-gray-500">CHORDS </span>
            <span className="font-medium text-gray-700">
              {uniqueChords.slice(0, 4).map(c => c.chord).join(' • ')}
            </span>
          </div>
          <div>
            <span className="text-gray-500">KEY </span>
            <span className="font-bold text-gray-900">{detectedKey}</span>
          </div>
        </div>
        <div className="text-gray-500">
          {uniqueChords.length} unique chords
        </div>
      </div>
    </div>
  );
};

export default ChordPlayer;
