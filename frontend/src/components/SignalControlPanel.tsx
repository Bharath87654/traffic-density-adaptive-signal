import React, { useState, useEffect } from 'react';
import { Settings, Play, Square, AlertCircle } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface SignalControlPanelProps {
  activeLane: string;
  countdown: number;
}

const SignalControlPanel: React.FC<SignalControlPanelProps> = ({ activeLane, countdown }) => {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = (countdown / 30) * 100; // Assuming 30s phase

  return (
    <div className="dashboard-card p-6 flex flex-col h-full">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-widest">SignalControlPanel</h2>
        <Settings className="w-4 h-4 text-slate-500 cursor-pointer hover:text-white transition-colors" />
      </div>

      <div className="bg-traffic-green/5 border border-traffic-green/20 rounded-lg p-4 mb-8 text-center">
        <span className="text-traffic-green text-sm font-bold tracking-[0.2em]">ACTIVE: {activeLane.toUpperCase()}</span>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center space-y-4">
        <div className="text-6xl font-['JetBrains_Mono'] neon-text-green font-bold tracking-tighter">
          {formatTime(countdown)}
        </div>
        <span className="text-slate-500 text-[10px] uppercase tracking-[0.3em]">Phase progress</span>
        
        <div className="w-full max-w-[200px] h-2 bg-slate-800 rounded-full overflow-hidden mt-2 relative">
            <div 
              className="h-full bg-traffic-green absolute left-0 top-0 transition-all duration-1000 ease-linear shadow-[0_0_10px_rgba(0,255,136,0.5)]"
              style={{ width: `${progress}%` }}
            />
        </div>
      </div>

      <div className="mt-8 space-y-6">
        <div className="flex justify-between items-center border-b border-traffic-border/50 pb-4">
          <span className="text-slate-400 text-sm">Vehicles passed</span>
          <span className="text-xl font-bold text-white">25</span>
        </div>

        <button className="w-full py-4 bg-[#1e293b] border border-traffic-border rounded-lg text-slate-300 font-bold uppercase tracking-widest text-xs hover:bg-slate-800 hover:border-traffic-red/50 hover:text-traffic-red transition-all group flex items-center justify-center gap-2">
          <AlertCircle className="w-4 h-4 group-hover:animate-pulse" />
          MANUAL OVERRIDE
        </button>
      </div>
    </div>
  );
};

export default SignalControlPanel;
