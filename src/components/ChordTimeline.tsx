import { useMemo } from 'react';
import { DetectedChord } from '@/services/chordDetector';
import { GuitarChord } from './GuitarChord';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Music } from 'lucide-react';

interface ChordTimelineProps {
  chords: DetectedChord[];
  currentTime: number;
  duration: number;
  onChordClick?: (time: number) => void;
  className?: string;
}

export const ChordTimeline = ({
  chords,
  currentTime,
  duration,
  onChordClick,
  className = '',
}: ChordTimelineProps) => {
  // Find the currently playing chord
  const activeChordIndex = useMemo(() => {
    return chords.findIndex(
      (chord) => currentTime >= chord.startTime && currentTime < chord.endTime
    );
  }, [chords, currentTime]);

  // Format time display
  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (chords.length === 0) {
    return (
      <div className={`glass-card p-8 text-center ${className}`}>
        <div className="p-4 rounded-full bg-muted/50 w-fit mx-auto mb-4">
          <Music className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="font-semibold text-lg mb-2">No chords detected yet</h3>
        <p className="text-muted-foreground">
          Upload an audio file or record your voice to see detected chords
        </p>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Current Chord Display */}
      <div className="glass-card p-6 text-center">
        <h3 className="text-sm font-medium text-muted-foreground mb-4">Now Playing</h3>
        {activeChordIndex >= 0 ? (
          <div className="flex flex-col items-center">
            <GuitarChord 
              chord={chords[activeChordIndex].chord} 
              size="lg" 
              isActive={true}
            />
            <div className="mt-4 flex items-center space-x-4 text-sm text-muted-foreground">
              <span>Confidence: {Math.round(chords[activeChordIndex].confidence * 100)}%</span>
              <span>•</span>
              <span>{formatTime(chords[activeChordIndex].startTime)} - {formatTime(chords[activeChordIndex].endTime)}</span>
            </div>
          </div>
        ) : (
          <div className="py-8">
            <p className="text-2xl font-bold text-muted-foreground">-</p>
            <p className="text-sm text-muted-foreground mt-2">Play audio to see active chord</p>
          </div>
        )}
      </div>

      {/* Progress Timeline */}
      <div className="glass-card p-4">
        <div className="relative h-3 bg-muted/50 rounded-full overflow-hidden mb-4">
          {/* Chord segments */}
          {chords.map((chord, index) => {
            const leftPercent = (chord.startTime / duration) * 100;
            const widthPercent = ((chord.endTime - chord.startTime) / duration) * 100;
            const isActive = index === activeChordIndex;
            
            return (
              <div
                key={index}
                className={`absolute h-full transition-colors cursor-pointer ${
                  isActive 
                    ? 'bg-primary' 
                    : 'bg-primary/40 hover:bg-primary/60'
                }`}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
                onClick={() => onChordClick?.(chord.startTime)}
                title={`${chord.chord} (${formatTime(chord.startTime)})`}
              />
            );
          })}
          
          {/* Current time indicator */}
          <div
            className="absolute top-0 w-0.5 h-full bg-white shadow-lg"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        </div>
        
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatTime(currentTime)}</span>
          <span>{chords.length} chords detected</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Chord Sequence Scroll */}
      <div className="glass-card p-4">
        <h3 className="text-sm font-medium text-muted-foreground mb-4">Chord Progression</h3>
        <ScrollArea className="w-full whitespace-nowrap">
          <div className="flex space-x-4 pb-4">
            {chords.map((chord, index) => (
              <div
                key={index}
                className={`flex-shrink-0 p-3 rounded-xl cursor-pointer transition-all duration-300 ${
                  index === activeChordIndex
                    ? 'bg-primary/20 border-2 border-primary scale-105'
                    : 'bg-white/5 border border-white/10 hover:bg-white/10'
                }`}
                onClick={() => onChordClick?.(chord.startTime)}
              >
                <GuitarChord 
                  chord={chord.chord} 
                  size="sm" 
                  isActive={index === activeChordIndex}
                />
                <div className="text-center mt-2">
                  <div className="text-xs text-muted-foreground">
                    {formatTime(chord.startTime)}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </div>

      {/* Full Chord List */}
      <div className="glass-card p-4">
        <h3 className="text-sm font-medium text-muted-foreground mb-4">All Detected Chords</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {chords.map((chord, index) => (
            <button
              key={index}
              className={`p-3 rounded-xl text-center transition-all ${
                index === activeChordIndex
                  ? 'bg-primary/20 border-2 border-primary'
                  : 'bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20'
              }`}
              onClick={() => onChordClick?.(chord.startTime)}
            >
              <div className={`text-lg font-bold ${
                index === activeChordIndex ? 'text-primary' : 'text-foreground'
              }`}>
                {chord.chord}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {formatTime(chord.startTime)}
              </div>
              <div className="text-xs text-muted-foreground/70">
                {Math.round(chord.confidence * 100)}%
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChordTimeline;
