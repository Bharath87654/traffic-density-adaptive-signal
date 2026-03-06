import React from 'react';
import type { Lane } from '../types/traffic';
import { Shield, Users, List, AlertTriangle } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface LaneCardProps {
  lane: Lane;
}

const LaneCard: React.FC<LaneCardProps> = ({ lane }) => {
  const isGreen = lane.signalStatus === 'GREEN';

  return (
    <div className={cn(
      "dashboard-card p-4 transition-all duration-500",
      lane.isActive ? "glow-border-green" : "glow-border-red"
    )}>
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-slate-300 font-bold tracking-wider">{lane.name}</h3>
        <div className={cn(
          "px-3 py-1 rounded border text-[10px] font-bold tracking-widest uppercase",
          isGreen ? "border-traffic-green text-traffic-green bg-traffic-green/10" : "border-traffic-red text-traffic-red bg-traffic-red/10"
        )}>
          [SIGNAL: {lane.signalStatus}]
        </div>
      </div>

      <div className="flex gap-4">
        {/* Video Placeholder */}
        <div className="relative w-1/2 aspect-video bg-black rounded-lg overflow-hidden border border-traffic-border">
          {lane.videoUrl?.endsWith('.mp4') ? (
            <video 
              src={lane.videoUrl} 
              autoPlay 
              loop 
              muted 
              className="w-full h-full object-cover opacity-60"
            />
          ) : (
            <img 
              src={lane.videoUrl} 
              alt={lane.name} 
              className="w-full h-full object-cover opacity-60"
            />
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          <div className="absolute bottom-2 left-2 flex items-center gap-1">
            <div className={cn("w-2 h-2 rounded-full", isGreen ? "bg-traffic-green animate-pulse" : "bg-traffic-red")} />
            <span className="text-[8px] text-white/70">LIVE FEED</span>
          </div>
        </div>

        {/* Metrics */}
        <div className="flex-1 space-y-2">
          <MetricRow label="VEHICLES" value={lane.vehicles} />
          <MetricRow label="QUEUE" value={lane.queue} />
          <MetricRow label="EMERGENCY" value={lane.emergency === 'NONE' ? 'NONE' : lane.emergency} isAlert={lane.emergency !== 'NONE'} />
          
          <div className="mt-4 pt-2 border-t border-traffic-border/50">
            <div className="flex justify-between text-[10px] text-slate-500 mb-1">
              <span>LOAD</span>
              <span>{Math.min(100, Math.round((lane.vehicles / 20) * 100))}%</span>
            </div>
            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={cn("h-full transition-all duration-1000", isGreen ? "bg-traffic-green" : "bg-traffic-yellow")}
                style={{ width: `${Math.min(100, Math.round((lane.vehicles / 20) * 100))}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricRow: React.FC<{ label: string; value: string | number; isAlert?: boolean }> = ({ label, value, isAlert }) => (
  <div className="flex justify-between items-center">
    <span className="text-[10px] text-slate-500 font-mono tracking-tighter">[{label}:]</span>
    <span className={cn(
      "text-sm font-bold font-mono",
      isAlert ? "text-traffic-red animate-pulse" : "text-slate-200"
    )}>{value}</span>
  </div>
);

export default LaneCard;
