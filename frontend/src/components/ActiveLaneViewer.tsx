import React from 'react';
import type { Detection } from '../types/traffic';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ActiveLaneViewerProps {
  videoUrl: string;
  detections: Detection[];
  realTimeCount: number;
  countdown: number;
}

const ActiveLaneViewer: React.FC<ActiveLaneViewerProps> = ({ videoUrl, detections, realTimeCount, countdown }) => {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="dashboard-card h-full relative overflow-hidden group">
      <div className="absolute top-4 left-4 z-10">
        <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-widest">ActiveLaneViewer</h2>
      </div>

      <div className="absolute top-4 right-4 z-10 bg-black/40 backdrop-blur-md border border-white/10 px-2 py-1 rounded text-[10px] text-slate-500">
        AI ASSISTED MONITORING
      </div>

      <div className="relative w-full h-full bg-black">
        {videoUrl.endsWith('.mp4') ? (
          <video 
            src={videoUrl} 
            autoPlay 
            loop 
            muted 
            className="w-full h-full object-cover opacity-80"
          />
        ) : (
          <img 
            src={videoUrl} 
            alt="Active Lane" 
            className="w-full h-full object-cover opacity-80"
          />
        )}
        
        {/* Detection Overlays */}
        <div className="absolute inset-0 pointer-events-none">
          {detections.map((det) => (
            <div 
              key={det.id}
              className="absolute border-2 border-traffic-green bg-traffic-green/10"
              style={{
                left: `${det.bbox[0]}%`,
                top: `${det.bbox[1]}%`,
                width: `${det.bbox[2]}%`,
                height: `${det.bbox[3]}%`,
              }}
            >
              <div className="absolute -top-5 left-0 bg-traffic-green px-1 py-0.5 text-[8px] font-bold text-black uppercase">
                {det.label} {Math.round(det.confidence * 100)}%
              </div>
            </div>
          ))}

          {/* Large Label Overlay Example from Reference */}
          <div className="absolute top-[20%] right-[10%] text-6xl font-bold neon-text-green opacity-40 select-none">
            CAR 92%
          </div>
        </div>

        {/* Info Overlays */}
        <div className="absolute bottom-6 left-6 flex flex-col items-start gap-1">
          <span className="text-slate-400 text-[10px] uppercase font-bold tracking-widest">Real Time Count</span>
          <span className="text-4xl font-bold neon-text-green">{realTimeCount.toLocaleString()}</span>
        </div>

        <div className="absolute bottom-6 right-6 flex flex-col items-end gap-1">
          <span className="text-slate-400 text-[10px] uppercase font-bold tracking-widest">Countdown Time</span>
          <div className="bg-[#1e293b] border border-traffic-border px-4 py-2 rounded-lg text-traffic-yellow text-2xl font-['JetBrains_Mono'] font-bold">
            {formatTime(countdown)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ActiveLaneViewer;
