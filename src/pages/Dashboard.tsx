import { useState, useRef, useEffect } from "react";
import { UserButton, useUser } from "@clerk/clerk-react";
import { Upload, Music, Mic, X, Loader2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ChordPlayer } from "@/components/ChordPlayer";
import { audioRecorder, RecordingResult } from "@/services/audioRecorder";
import { chordDetector, DetectedChord, ChordDetectionResult } from "@/services/chordDetector";
import { useSongLibrary, Song } from "@/hooks/useSongLibrary";
import { toast } from "@/hooks/use-toast";

type ViewState = 'home' | 'upload' | 'processing' | 'player';

const Dashboard = () => {
  const { user } = useUser();
  const [view, setView] = useState<ViewState>('home');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [currentSong, setCurrentSong] = useState<{
    name: string;
    audioUrl: string;
    chords: DetectedChord[];
  } | null>(null);

  const { songs, addSong } = useSongLibrary();

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      audioRecorder.cleanup();
    };
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleFile = async (file: File) => {
    if (!file.type.startsWith('audio/')) {
      toast({ title: "Invalid file", description: "Please upload an audio file", variant: "destructive" });
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast({ title: "File too large", description: "Max 50MB", variant: "destructive" });
      return;
    }

    setView('processing');
    setProcessingProgress(10);

    try {
      const audioUrl = URL.createObjectURL(file);
      setProcessingProgress(30);
      const result: ChordDetectionResult = await chordDetector.processAudioFile(file);
      setProcessingProgress(80);
      const songName = file.name.replace(/\.[^/.]+$/, "");
      await addSong(songName, result.chords, file, 'upload', result.duration);
      setCurrentSong({ name: songName, audioUrl, chords: result.chords });
      setProcessingProgress(100);
      setTimeout(() => setView('player'), 300);
    } catch {
      toast({ title: "Error", description: "Failed to analyze audio", variant: "destructive" });
      setView('upload');
    }
  };

  const startRecording = async () => {
    try {
      await audioRecorder.startRecording();
      setIsRecording(true);
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
    } catch {
      toast({ title: "Microphone denied", description: "Please allow microphone access", variant: "destructive" });
    }
  };

  const stopRecording = async () => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    setIsRecording(false);
    setView('processing');
    setProcessingProgress(20);

    try {
      const result: RecordingResult = await audioRecorder.stopRecording();
      setProcessingProgress(50);
      const chordResult: ChordDetectionResult = await chordDetector.processAudioBlob(result.blob);
      setProcessingProgress(80);
      const songName = `Recording ${new Date().toLocaleTimeString()}`;
      await addSong(songName, chordResult.chords, result.blob, 'recording', result.duration);
      setCurrentSong({ name: songName, audioUrl: result.url, chords: chordResult.chords });
      setProcessingProgress(100);
      setTimeout(() => { setView('player'); setRecordingTime(0); }, 300);
    } catch {
      toast({ title: "Error", description: "Failed to process recording", variant: "destructive" });
      setView('upload');
    }
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  const openSong = (song: Song) => {
    setCurrentSong({ name: song.name, audioUrl: song.audioUrl, chords: song.chords });
    setView('player');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {view !== 'home' && (
              <button onClick={() => setView('home')} className="p-2 text-gray-500 hover:text-gray-700">
                <ArrowLeft className="h-5 w-5" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded bg-[#00c7b7]">
                <Music className="h-4 w-4 text-white" />
              </div>
              <span className="font-bold text-gray-900">ChordAI</span>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {view === 'home' && (
              <Button size="sm" onClick={() => setView('upload')} className="bg-[#00c7b7] hover:bg-[#00b5a6]">
                <Upload className="h-4 w-4 mr-1" /> Upload
              </Button>
            )}
            <UserButton />
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6">
        
        {/* Home */}
        {view === 'home' && (
          <div className="space-y-6">
            <div className="text-center py-8">
              <h1 className="text-2xl font-bold text-gray-900 mb-2">
                Welcome{user?.firstName ? `, ${user.firstName}` : ''}!
              </h1>
              <p className="text-gray-500 mb-6">Upload audio or record to detect guitar chords</p>
              
              <div className="flex gap-3 justify-center">
                <Button onClick={() => setView('upload')} className="bg-[#00c7b7] hover:bg-[#00b5a6]">
                  <Upload className="h-4 w-4 mr-2" /> Upload Audio
                </Button>
                <Button variant="outline" onClick={() => setView('upload')}>
                  <Mic className="h-4 w-4 mr-2" /> Record
                </Button>
              </div>
            </div>

            {songs.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold text-gray-700 mb-3">Recent Songs</h2>
                <div className="grid gap-2">
                  {songs.slice(0, 5).map((song) => (
                    <button
                      key={song.id}
                      onClick={() => openSong(song)}
                      className="flex items-center gap-3 p-3 bg-white rounded-lg border border-gray-200 hover:border-[#00c7b7] transition-colors text-left"
                    >
                      <div className="p-2 bg-gray-100 rounded">
                        {song.type === 'recording' ? <Mic className="h-4 w-4 text-gray-600" /> : <Music className="h-4 w-4 text-gray-600" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 truncate text-sm">{song.name}</p>
                        <p className="text-xs text-gray-500">{song.chords.length} chords • {Math.round(song.duration)}s</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Upload */}
        {view === 'upload' && (
          <div className="max-w-md mx-auto space-y-4">
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Analyze Audio</h2>
              <p className="text-gray-500 text-sm">Upload a file or record</p>
            </div>

            {/* Drop Zone */}
            <div
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                dragActive ? 'border-[#00c7b7] bg-[#00c7b7]/5' : 'border-gray-300 hover:border-gray-400 bg-white'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                className="hidden"
              />
              <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-700 font-medium text-sm">Drop audio file here</p>
              <p className="text-xs text-gray-500">or click to browse • MP3, WAV up to 50MB</p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex-1 h-px bg-gray-200" />
              <span className="text-xs text-gray-400">OR</span>
              <div className="flex-1 h-px bg-gray-200" />
            </div>

            {/* Recording */}
            <div className="bg-white rounded-xl p-6 text-center border border-gray-200">
              <div className={`inline-flex p-4 rounded-full mb-3 ${isRecording ? 'bg-red-100' : 'bg-gray-100'}`}>
                <Mic className={`h-6 w-6 ${isRecording ? 'text-red-500' : 'text-gray-500'}`} />
              </div>
              
              {isRecording && (
                <div className="mb-3">
                  <p className="text-red-500 font-bold text-xl">{formatTime(recordingTime)}</p>
                  <p className="text-xs text-gray-500">Recording...</p>
                </div>
              )}
              
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                className={isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-[#00c7b7] hover:bg-[#00b5a6]'}
              >
                {isRecording ? <><X className="h-4 w-4 mr-1" /> Stop & Analyze</> : <><Mic className="h-4 w-4 mr-1" /> Start Recording</>}
              </Button>
            </div>
          </div>
        )}

        {/* Processing */}
        {view === 'processing' && (
          <div className="max-w-sm mx-auto text-center py-12">
            <Loader2 className="h-10 w-10 text-[#00c7b7] animate-spin mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-gray-900 mb-1">Analyzing...</h2>
            <p className="text-gray-500 text-sm mb-4">Detecting chords</p>
            <Progress value={processingProgress} className="h-1.5" />
          </div>
        )}

        {/* Player */}
        {view === 'player' && currentSong && (
          <div className="space-y-4">
            <ChordPlayer
              audioUrl={currentSong.audioUrl}
              chords={currentSong.chords}
              songName={currentSong.name}
            />
            
            <div className="text-center">
              <Button variant="outline" size="sm" onClick={() => setView('upload')}>
                <Upload className="h-4 w-4 mr-1" /> Analyze Another
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
