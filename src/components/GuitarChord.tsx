import { useMemo } from 'react';

interface GuitarChordProps {
  chord: string;
  size?: 'sm' | 'md' | 'lg';
  isActive?: boolean;
}

const CHORD_DB: Record<string, { positions: number[]; fingers: number[] }> = {
  'C': { positions: [-1, 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0] },
  'Cm': { positions: [-1, 3, 5, 5, 4, 3], fingers: [0, 1, 3, 4, 2, 1] },
  'C7': { positions: [-1, 3, 2, 3, 1, 0], fingers: [0, 3, 2, 4, 1, 0] },
  'D': { positions: [-1, -1, 0, 2, 3, 2], fingers: [0, 0, 0, 1, 3, 2] },
  'Dm': { positions: [-1, -1, 0, 2, 3, 1], fingers: [0, 0, 0, 2, 3, 1] },
  'D7': { positions: [-1, -1, 0, 2, 1, 2], fingers: [0, 0, 0, 2, 1, 3] },
  'E': { positions: [0, 2, 2, 1, 0, 0], fingers: [0, 2, 3, 1, 0, 0] },
  'Em': { positions: [0, 2, 2, 0, 0, 0], fingers: [0, 2, 3, 0, 0, 0] },
  'E7': { positions: [0, 2, 0, 1, 0, 0], fingers: [0, 2, 0, 1, 0, 0] },
  'F': { positions: [1, 3, 3, 2, 1, 1], fingers: [1, 3, 4, 2, 1, 1] },
  'Fm': { positions: [1, 3, 3, 1, 1, 1], fingers: [1, 3, 4, 1, 1, 1] },
  'G': { positions: [3, 2, 0, 0, 0, 3], fingers: [2, 1, 0, 0, 0, 3] },
  'Gm': { positions: [3, 5, 5, 3, 3, 3], fingers: [1, 3, 4, 1, 1, 1] },
  'G7': { positions: [3, 2, 0, 0, 0, 1], fingers: [3, 2, 0, 0, 0, 1] },
  'A': { positions: [-1, 0, 2, 2, 2, 0], fingers: [0, 0, 1, 2, 3, 0] },
  'Am': { positions: [-1, 0, 2, 2, 1, 0], fingers: [0, 0, 2, 3, 1, 0] },
  'A7': { positions: [-1, 0, 2, 0, 2, 0], fingers: [0, 0, 1, 0, 2, 0] },
  'B': { positions: [-1, 2, 4, 4, 4, 2], fingers: [0, 1, 2, 3, 4, 1] },
  'Bm': { positions: [-1, 2, 4, 4, 3, 2], fingers: [0, 1, 3, 4, 2, 1] },
  'B7': { positions: [-1, 2, 1, 2, 0, 2], fingers: [0, 2, 1, 3, 0, 4] },
};

export const GuitarChord = ({ chord, size = 'md', isActive = false }: GuitarChordProps) => {
  const data = useMemo(() => {
    if (CHORD_DB[chord]) return CHORD_DB[chord];
    const root = chord.match(/^[A-G][#b]?/)?.[0];
    if (root) {
      const isMinor = chord.toLowerCase().includes('m') && !chord.toLowerCase().includes('maj');
      const key = isMinor ? `${root}m` : root;
      if (CHORD_DB[key]) return CHORD_DB[key];
      if (CHORD_DB[root]) return CHORD_DB[root];
    }
    return null;
  }, [chord]);

  const cfg = {
    sm: { w: 50, h: 65, dot: 4, font: 10 },
    md: { w: 70, h: 85, dot: 5, font: 12 },
    lg: { w: 90, h: 105, dot: 6, font: 14 },
  }[size];

  const strGap = (cfg.w - 12) / 5;
  const fretGap = (cfg.h - 24) / 4;
  const sx = 6;
  const sy = 14;

  if (!data) {
    return (
      <div className="flex flex-col items-center">
        <div 
          className="border border-gray-300 rounded bg-white flex items-center justify-center"
          style={{ width: cfg.w, height: cfg.h }}
        >
          <span className="text-gray-400 text-xs">?</span>
        </div>
        <span className="mt-1 text-xs font-medium text-gray-600">{chord}</span>
      </div>
    );
  }

  const { positions, fingers } = data;
  const minFret = Math.min(...positions.filter(p => p > 0));
  const maxFret = Math.max(...positions);
  const startFret = maxFret > 4 ? minFret : 1;

  return (
    <div className="flex flex-col items-center">
      <svg width={cfg.w} height={cfg.h} className="bg-white border border-gray-200 rounded">
        {/* Nut or fret number */}
        {startFret === 1 ? (
          <rect x={sx} y={sy} width={strGap * 5} height={2} fill="#333" />
        ) : (
          <text x={2} y={sy + 8} fontSize="8" fill="#666">{startFret}</text>
        )}

        {/* Frets */}
        {[0, 1, 2, 3, 4].map(f => (
          <line key={f} x1={sx} y1={sy + f * fretGap} x2={sx + strGap * 5} y2={sy + f * fretGap} stroke="#ccc" strokeWidth={1} />
        ))}

        {/* Strings */}
        {[0, 1, 2, 3, 4, 5].map(s => (
          <line key={s} x1={sx + s * strGap} y1={sy} x2={sx + s * strGap} y2={sy + fretGap * 4} stroke="#999" strokeWidth={1} />
        ))}

        {/* Finger positions */}
        {positions.map((fret, str) => {
          const x = sx + str * strGap;
          
          if (fret === -1) {
            return <text key={str} x={x} y={sy - 3} textAnchor="middle" fontSize="9" fill="#666">×</text>;
          }
          
          if (fret === 0) {
            return <circle key={str} cx={x} cy={sy - 5} r={3} fill="none" stroke="#666" strokeWidth={1.5} />;
          }
          
          const displayFret = fret - startFret + 1;
          const y = sy + (displayFret - 0.5) * fretGap;
          
          return (
            <g key={str}>
              <circle cx={x} cy={y} r={cfg.dot} fill={isActive ? '#00c7b7' : '#333'} />
              {fingers[str] > 0 && (
                <text x={x} y={y + 3} textAnchor="middle" fontSize="8" fill="white" fontWeight="bold">
                  {fingers[str]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      
      <span className={`mt-1 font-semibold ${isActive ? 'text-[#00c7b7]' : 'text-gray-700'}`} style={{ fontSize: cfg.font }}>
        {chord}
      </span>
    </div>
  );
};

export default GuitarChord;
