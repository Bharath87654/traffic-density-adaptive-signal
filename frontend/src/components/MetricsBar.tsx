import React from 'react';
import type { SystemMetrics } from '../types/traffic';
import { Activity, Clock, Zap, TrendingUp } from 'lucide-react';

interface MetricsBarProps {
  metrics: SystemMetrics;
}

const MetricsBar: React.FC<MetricsBarProps> = ({ metrics }) => {
  return (
    <div className="flex gap-4 p-4 mt-auto">
      <MetricCard 
        label="Total Detected Today" 
        value={metrics.totalDetectedToday.toLocaleString()} 
        icon={<Activity className="w-4 h-4 text-slate-400" />}
      />
      <MetricCard 
        label="Avg Wait" 
        value={metrics.avgWaitTime} 
        icon={<Clock className="w-4 h-4 text-traffic-yellow" />}
        valueClass="text-traffic-yellow"
      />
      <MetricCard 
        label="AI Switches" 
        value={metrics.aiSwitches.toString()} 
        icon={<Zap className="w-4 h-4 text-slate-400" />}
      />
      <MetricCard 
        label="Peak Hour" 
        value={metrics.peakHour} 
        icon={<TrendingUp className="w-4 h-4 text-slate-400" />}
      />
    </div>
  );
};

const MetricCard: React.FC<{ label: string; value: string; icon: React.ReactNode; valueClass?: string }> = ({ 
  label, value, icon, valueClass = "text-white" 
}) => (
  <div className="flex-1 dashboard-card flex items-center justify-between p-3 px-6 hover:border-traffic-green/30 transition-colors">
    <div className="flex items-center gap-3">
      {icon}
      <span className="text-slate-400 text-sm font-medium">{label}</span>
    </div>
    <span className={`text-xl font-bold ${valueClass} neon-text-green`}>{value}</span>
  </div>
);

export default MetricsBar;
