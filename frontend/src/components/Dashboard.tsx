import React, { useState, useEffect } from 'react';
import type { Lane, AnalyticsData, SystemMetrics, Detection } from '../types/traffic';
import { trafficApi } from '../services/api';
import LaneCard from './LaneCard';
import SignalControlPanel from './SignalControlPanel';
import ActiveLaneViewer from './ActiveLaneViewer';
import AnalyticsPanel from './AnalyticsPanel';
import MetricsBar from './MetricsBar';
import { Shield, Settings, User } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [lanes, setLanes] = useState<Lane[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [countdown, setCountdown] = useState(8);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [lanesData, analyticsData, metricsData] = await Promise.all([
          trafficApi.fetchLanes(),
          trafficApi.fetchAnalytics(),
          trafficApi.fetchSystemMetrics()
        ]);
        setLanes(lanesData);
        setAnalytics(analyticsData);
        setMetrics(metricsData);
      } catch (error) {
        console.error("Failed to fetch dashboard data", error);
      } finally {
        setLoading(false);
      }
    };

    loadData();

    // Simulating real-time countdown
    const timer = setInterval(() => {
      setCountdown(prev => (prev > 0 ? prev - 1 : 30));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const activeLane = lanes.find(l => l.isActive) || lanes[0];

  const mockDetections: Detection[] = [
    { id: '1', label: 'CAR', confidence: 0.92, bbox: [20, 30, 25, 20] },
    { id: '2', label: 'BUS', confidence: 0.88, bbox: [50, 40, 30, 40] },
    { id: '3', label: 'TRUCK', confidence: 0.91, bbox: [10, 60, 20, 30] },
  ];

  if (loading || !metrics || !analytics) {
    return (
      <div className="flex items-center justify-center h-screen bg-traffic-navy text-traffic-green font-mono">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-traffic-green border-t-transparent rounded-full animate-spin" />
          <span className="animate-pulse tracking-widest uppercase text-sm">Initializing Command Center...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-traffic-navy overflow-hidden flex flex-col text-slate-200 p-2">
      {/* Header (Optional but adds to center dashboard feel) */}
      <header className="flex justify-between items-center px-4 py-2 border-b border-traffic-border/30 mb-2">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-traffic-green/20 border border-traffic-green/40 flex items-center justify-center">
            <Shield className="w-5 h-5 text-traffic-green" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tighter text-white">SMART TRAFFIC <span className="text-traffic-green">CONTROL</span></h1>
            <p className="text-[8px] text-slate-500 uppercase tracking-widest font-bold">Autonomous Intersection Management System</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">System Status</span>
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-traffic-green animate-pulse" />
              <span className="text-[10px] text-traffic-green font-bold">OPERATIONAL</span>
            </div>
          </div>
          <div className="w-px h-8 bg-traffic-border/50" />
          <div className="flex items-center gap-4">
            <Settings className="w-5 h-5 text-slate-400 cursor-pointer hover:text-white transition-colors" />
            <div className="flex items-center gap-2 bg-slate-800/50 p-1 pr-3 rounded-full border border-traffic-border cursor-pointer">
              <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center">
                <User className="w-4 h-4 text-slate-300" />
              </div>
              <span className="text-xs font-medium text-slate-300">ADMIN</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 grid grid-rows-[auto_1fr] gap-3 px-2">
        {/* Top Section - Lane Grid */}
        <div className="grid grid-cols-2 gap-3 h-[30vh]">
          {lanes.map(lane => (
            <LaneCard key={lane.id} lane={lane} />
          ))}
        </div>

        {/* Bottom Section - Controls & Analytics */}
        <div className="grid grid-cols-[1fr_1.8fr_1.2fr] gap-3 h-[52vh]">
          <SignalControlPanel 
            activeLane={activeLane.name} 
            countdown={countdown} 
          />
          <ActiveLaneViewer 
            videoUrl={activeLane.videoUrl || ''} 
            detections={mockDetections}
            realTimeCount={metrics.totalDetectedToday}
            countdown={countdown}
          />
          <AnalyticsPanel data={analytics} />
        </div>
      </main>

      {/* Bottom Metrics Bar */}
      <MetricsBar metrics={metrics} />
    </div>
  );
};

export default Dashboard;
