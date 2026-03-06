import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import {
  AlertTriangle, Car, Clock, Activity, Zap, ShieldAlert,
  Video, Radio, Server, CheckCircle2, Cpu
} from 'lucide-react';

// ==========================================
// TYPES & INTERFACES
// ==========================================

export interface Lane {
  id: number;
  vehicles: number;
  queue: number;
  signal: 'green' | 'yellow' | 'red';
  emergency: boolean;
}

export interface Detection {
  type: string;
  confidence: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface VolumeData {
  time: string;
  lane1: number;
  lane2: number;
  lane3?: number;
  lane4?: number;
}

export interface WaitTimeData {
  name: string;
  wait: number;
}

export interface EmergencyData {
  name: string;
  value: number;
}

export interface PassedData {
  cycle: string;
  count: number;
}

export interface AnalyticsData {
  volume: VolumeData[];
  waitTimes: WaitTimeData[];
  emergency: EmergencyData[];
  passed: PassedData[];
}

// ==========================================
// MOCK DATA
// ==========================================

const MOCK_ANALYTICS: AnalyticsData = {
  volume: [
    { time: '08:00', lane1: 45, lane2: 30, lane3: 20, lane4: 15 },
    { time: '09:00', lane1: 80, lane2: 50, lane3: 35, lane4: 25 },
    { time: '10:00', lane1: 65, lane2: 40, lane3: 30, lane4: 20 },
    { time: '11:00', lane1: 50, lane2: 35, lane3: 25, lane4: 15 },
    { time: '12:00', lane1: 55, lane2: 45, lane3: 35, lane4: 30 },
  ],
  waitTimes: [
    { name: 'Lane 1', wait: 24 },
    { name: 'Lane 2', wait: 35 },
    { name: 'Lane 3', wait: 18 },
    { name: 'Lane 4', wait: 12 },
  ],
  emergency: [
    { name: 'Ambulance', value: 12 },
    { name: 'Firetruck', value: 4 },
    { name: 'Police', value: 8 },
  ],
  passed: [
    { cycle: 'C-1', count: 120 },
    { cycle: 'C-2', count: 95 },
    { cycle: 'C-3', count: 140 },
    { cycle: 'C-4', count: 110 },
  ]
};

// ==========================================
// HOOKS
// ==========================================

function useTrafficData() {
  const [lanes, setLanes] = useState<Lane[]>([
    { id: 1, vehicles: 0, queue: 0, signal: 'red', emergency: false },
    { id: 2, vehicles: 0, queue: 0, signal: 'red', emergency: false },
    { id: 3, vehicles: 0, queue: 0, signal: 'red', emergency: false },
    { id: 4, vehicles: 0, queue: 0, signal: 'red', emergency: false }
  ]);
  const [activeLane, setActiveLane] = useState<number>(1);
  const [countdown, setCountdown] = useState<number>(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData>(MOCK_ANALYTICS);
  const [isAIActive, setIsAIActive] = useState<boolean>(true);

  const activeLaneRef = useRef<number>(activeLane);

  useEffect(() => {
    activeLaneRef.current = activeLane;
  }, [activeLane]);

  // REST API Polling
  useEffect(() => {
    const fetchRestData = async () => {
      try {
        const [lanesRes, analyticsRes] = await Promise.all([
          fetch('http://localhost:8000/api/lanes').catch(() => null),
          fetch('http://localhost:8000/api/traffic-analytics').catch(() => null)
        ]);

        if (lanesRes && lanesRes.ok) {
          const data = await lanesRes.json();
          setLanes(data.lanes || data || []);
        }

        if (analyticsRes && analyticsRes.ok) {
          const data = await analyticsRes.json();
          if (data && Object.keys(data).length > 0) {
            setAnalytics(data);
          }
        }
      } catch (error) {
        console.error("Failed to fetch REST data:", error);
      }
    };

    fetchRestData();
    const restInterval = setInterval(fetchRestData, 2000);
    return () => clearInterval(restInterval);
  }, []);

  // WebSocket: Signal Status
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/signal-status');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.active_lane !== undefined) setActiveLane(data.active_lane);
        if (data.countdown !== undefined) setCountdown(data.countdown);
      } catch (e) {
        console.error("Error parsing signal WS message", e);
      }
    };

    ws.onerror = (error) => console.error("Signal WS Error:", error);

    return () => {
      if (ws.readyState === 1) ws.close();
    };
  }, []);

  // WebSocket: Vehicle Detections
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/vehicle-detection');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.lane === activeLaneRef.current && data.detections) {
          setDetections(data.detections);
        }
      } catch (e) {
        console.error("Error parsing detection WS message", e);
      }
    };

    ws.onerror = (error) => console.error("Detection WS Error:", error);

    return () => {
      if (ws.readyState === 1) ws.close();
    };
  }, []);

  const toggleAIControl = async () => {
    const newState = !isAIActive;
    setIsAIActive(newState);
    try {
      await fetch('http://localhost:8000/api/signal/ai-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: newState })
      });
    } catch (error) {
      console.error("Failed to toggle AI control:", error);
    }
  };

  return { lanes, activeLane, countdown, detections, analytics, toggleAIControl, isAIActive };
}


// ==========================================
// COMPONENTS
// ==========================================

interface LaneCardProps {
  lane: Lane;
  isActive: boolean;
}

const LaneCard: React.FC<LaneCardProps> = ({ lane, isActive }) => {
  return (
    <div className={`relative flex flex-col bg-gray-900 rounded-xl overflow-hidden border transition-all duration-300 ${
      isActive
        ? 'border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.3)]'
        : 'border-slate-800'
    }`}>
      <div className="flex justify-between items-center px-4 py-2 bg-slate-800/50 border-b border-slate-700/50">
        <h3 className="text-slate-200 font-semibold flex items-center gap-2">
          <Video size={16} className="text-slate-400" />
          Lane {lane.id} - {lane.id === 1 ? 'North' : lane.id === 2 ? 'East' : lane.id === 3 ? 'South' : 'West'} Bound
        </h3>
        <div className="flex items-center gap-3">
          {lane.emergency && (
            <span className="flex items-center gap-1 text-xs font-bold text-red-500 animate-pulse bg-red-500/10 px-2 py-1 rounded-full border border-red-500/20">
              <AlertTriangle size={12} /> EMERGENCY
            </span>
          )}
          <div className="flex gap-1">
            <div className={`w-3 h-3 rounded-full ${lane.signal === 'red' ? 'bg-red-500 shadow-[0_0_10px_#ef4444]' : 'bg-slate-700'}`} />
            <div className={`w-3 h-3 rounded-full ${lane.signal === 'yellow' ? 'bg-yellow-400 shadow-[0_0_10px_#facc15]' : 'bg-slate-700'}`} />
            <div className={`w-3 h-3 rounded-full ${lane.signal === 'green' ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-slate-700'}`} />
          </div>
        </div>
      </div>

      <div className="relative h-48 bg-black flex items-center justify-center overflow-hidden">
        <video
          src={`/lane${lane.id}.mp4`}
          autoPlay
          loop
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover opacity-80"
        />
        <div className="absolute bottom-0 w-full h-1/2 bg-gradient-to-t from-slate-900 to-transparent z-10 pointer-events-none" />
        <div className="absolute top-2 right-2 flex items-center gap-2 z-10">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
          </span>
          <span className="text-[10px] text-slate-400 font-mono shadow-black drop-shadow-md">LIVE CAM-0{lane.id}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 divide-x divide-slate-800 bg-slate-900/50">
        <div className="p-3 flex items-center justify-between">
          <div className="text-slate-400 text-xs uppercase tracking-wider">Volume</div>
          <div className="text-xl font-bold text-white flex items-center gap-2">
            <Car size={16} className="text-blue-400" />
            {lane.vehicles}
          </div>
        </div>
        <div className="p-3 flex items-center justify-between">
          <div className="text-slate-400 text-xs uppercase tracking-wider">Queue</div>
          <div className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={16} className={lane.queue > 5 ? 'text-red-400' : 'text-green-400'} />
            {lane.queue}
          </div>
        </div>
      </div>
    </div>
  );
};

interface SignalControlPanelProps {
  activeLane: number;
  countdown: number;
  onToggleAI: () => void;
  isAIActive: boolean;
}

const SignalControlPanel: React.FC<SignalControlPanelProps> = ({ activeLane, countdown, onToggleAI, isAIActive }) => {
  const progress = Math.max(0, Math.min(100, ((15 - countdown) / 15) * 100));

  return (
    <div className="bg-gray-900 rounded-xl border border-slate-800 p-5 flex flex-col justify-between h-full">
      <div>
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-lg font-bold text-white mb-1">Signal Control Matrix</h2>
            <p className="text-sm text-slate-400">Automated Phase Sequencing</p>
          </div>
          <div className="bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-2">
            <Radio size={14} className="text-green-400 animate-pulse" />
            <span className="text-xs font-mono text-green-400">WS CONNECTED</span>
          </div>
        </div>

        <div className="flex items-center justify-between mb-8">
          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2 uppercase tracking-wider">Active Phase</div>
            <div className="text-4xl font-bold text-green-400 shadow-green-500/20 drop-shadow-md">
              LANE {activeLane}
            </div>
          </div>

          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2 uppercase tracking-wider">Time Remaining</div>
            <div className="text-5xl font-mono font-bold text-white flex items-baseline justify-center gap-1">
              {countdown.toString().padStart(2, '0')}
              <span className="text-xl text-slate-500">s</span>
            </div>
          </div>
        </div>

        <div className="space-y-2 mb-6">
          <div className="flex justify-between text-xs text-slate-400 font-mono">
            <span>PHASE PROGRESS</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all duration-1000 ease-linear"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      <button
        onClick={onToggleAI}
        className={`w-full py-3 rounded-lg font-bold tracking-wide flex items-center justify-center gap-2 transition-all duration-300 ${
          isAIActive
            ? 'bg-blue-500/10 text-blue-400 border border-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.2)] hover:bg-blue-500/20'
            : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700 hover:text-white'
        }`}
      >
        <Cpu size={18} className={isAIActive ? "animate-pulse" : ""} />
        {isAIActive ? 'AI OPTIMIZATION ACTIVE' : 'ENABLE AI CONTROL'}
      </button>
    </div>
  );
};

interface ActiveLaneViewerProps {
  laneId: number;
  detections: Detection[];
  countdown?: number; // Kept for future overlay extensions
}

const ActiveLaneViewer: React.FC<ActiveLaneViewerProps> = ({ laneId, detections }) => {
  return (
    <div className="bg-gray-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 bg-slate-800/50 border-b border-slate-700/50 flex justify-between items-center">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <Activity size={16} className="text-green-400" />
          AI Vision: Lane {laneId} Analysis
        </h2>
        <span className="text-xs bg-green-500/10 text-green-400 border border-green-500/20 px-2 py-1 rounded">
          {detections.length} OBJECTS DETECTED
        </span>
      </div>

      <div className="flex-1 relative bg-black overflow-hidden group">
        <video
          src={`/lane${laneId}.mp4`}
          autoPlay
          loop
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover opacity-70"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-transparent to-transparent pointer-events-none z-0" />

        {detections.map((det, i) => (
          <div
            key={i}
            className="absolute border-2 border-green-500 bg-green-500/10 transition-all duration-300 z-10"
            style={{
              left: `${det.x}%`,
              top: `${det.y}%`,
              width: `${det.w}%`,
              height: `${det.h}%`,
              boxShadow: '0 0 10px rgba(34, 197, 94, 0.4) inset'
            }}
          >
            <div className="absolute -top-5 left-[-2px] bg-green-500 text-black text-[10px] font-bold px-1 whitespace-nowrap">
              {det.type.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}

        <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end pointer-events-none">
          <div className="bg-black/60 backdrop-blur-sm border border-slate-700/50 p-2 rounded text-xs font-mono text-slate-300">
            FPS: 59.94 <br />
            RES: 1080P <br />
            LAT: 12ms
          </div>
          <div className="text-6xl font-black text-white/20">
            0{laneId}
          </div>
        </div>
      </div>
    </div>
  );
};

interface AnalyticsPanelProps {
  data: AnalyticsData;
}

const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({ data }) => {
  const PIE_COLORS = ['#ef4444', '#f97316', '#3b82f6'];

  return (
    <div className="bg-gray-900 rounded-xl border border-slate-800 p-4 h-full flex flex-col">
      <h2 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
        <Server size={16} className="text-blue-400" />
        System Analytics
      </h2>

      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-4">
        <div className="bg-slate-950/50 rounded-lg p-2 border border-slate-800/50 relative">
          <div className="text-[10px] text-slate-400 absolute top-2 left-2 z-10 uppercase">Traffic Volume</div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.volume} margin={{ top: 20, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '12px' }} />
              <Line type="monotone" dataKey="lane1" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="lane2" stroke="#22c55e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-950/50 rounded-lg p-2 border border-slate-800/50 relative">
          <div className="text-[10px] text-slate-400 absolute top-2 left-2 z-10 uppercase">Avg Wait (s)</div>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.waitTimes} margin={{ top: 20, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="name" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', border: 'none', fontSize: '12px' }} />
              <Bar dataKey="wait" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-950/50 rounded-lg p-2 border border-slate-800/50 relative flex items-center justify-center">
          <div className="text-[10px] text-slate-400 absolute top-2 left-2 z-10 uppercase">Emergency EVs</div>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data.emergency} cx="50%" cy="50%" innerRadius={25} outerRadius={40} paddingAngle={5} dataKey="value">
                {data.emergency.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '12px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-950/50 rounded-lg p-2 border border-slate-800/50 relative">
          <div className="text-[10px] text-slate-400 absolute top-2 left-2 z-10 uppercase">Cycle Thru-put</div>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.passed} margin={{ top: 20, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="cycle" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', border: 'none', fontSize: '12px' }} />
              <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const SystemMetricsBar: React.FC = () => {
  return (
    <div className="bg-gray-900 border-t border-slate-800 px-6 py-3 flex items-center justify-between text-sm">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-slate-400">
          <CheckCircle2 size={16} className="text-green-500" />
          <span>System Status: <span className="text-white font-medium">OPTIMAL</span></span>
        </div>
        <div className="w-px h-4 bg-slate-700" />
        <div className="flex items-center gap-2 text-slate-400">
          <Car size={16} />
          <span>Total Today: <span className="text-white font-mono">14,285</span></span>
        </div>
        <div className="w-px h-4 bg-slate-700" />
        <div className="flex items-center gap-2 text-slate-400">
          <Clock size={16} />
          <span>Avg Wait: <span className="text-white font-mono">22.4s</span></span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-slate-400">
          <Zap size={16} className="text-yellow-500" />
          <span>AI Switches: <span className="text-white font-mono">1,042</span></span>
        </div>
        <div className="w-px h-4 bg-slate-700" />
        <div className="flex items-center gap-2 text-slate-400">
          <ShieldAlert size={16} className="text-red-400" />
          <span>Peak Hour: <span className="text-white font-medium">08:00 - 09:00</span></span>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// MAIN APP COMPONENT
// ==========================================

export default function App() {
  const {
    lanes,
    activeLane,
    countdown,
    detections,
    analytics,
    toggleAIControl,
    isAIActive
  } = useTrafficData();

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans flex flex-col overflow-hidden selection:bg-green-500/30">

      <header className="px-6 py-4 border-b border-slate-800 bg-gray-900 flex justify-between items-center shadow-md z-10">
        <div className="flex items-center gap-3">
          <div className="bg-green-500/20 p-2 rounded-lg border border-green-500/30">
            <Activity className="text-green-400" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-white shadow-sm">
              NEXUS <span className="text-green-400 font-light">TrafficControl</span>
            </h1>
            <p className="text-xs text-slate-400 font-mono tracking-widest mt-0.5">ZONE-7 INTERSECTION ALPHA</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_#22c55e]" />
            <span className="text-xs font-medium text-slate-300">API ACTIVE</span>
          </div>
          <div className="text-sm font-mono text-slate-400 bg-slate-800 px-3 py-1.5 rounded-lg">
            {new Date().toLocaleTimeString('en-US', { hour12: false })}
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 grid grid-rows-[auto_1fr] gap-6 overflow-hidden">

        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <Video size={16} /> Live Feed Matrix
            </h2>
            <div className="text-xs text-slate-500 font-mono">
              LATENCY: 12ms | REST POLLING: 2s
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {lanes.map(lane => (
              <LaneCard key={lane.id} lane={lane} isActive={lane.id === activeLane} />
            ))}
          </div>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-0">
          <div className="h-[350px] lg:h-auto">
             <SignalControlPanel
                activeLane={activeLane}
                countdown={countdown}
                onToggleAI={toggleAIControl}
                isAIActive={isAIActive}
             />
          </div>
          <div className="h-[350px] lg:h-auto">
             <ActiveLaneViewer
                laneId={activeLane}
                detections={detections}
                countdown={countdown}
             />
          </div>
          <div className="h-[350px] lg:h-auto">
             <AnalyticsPanel data={analytics} />
          </div>
        </section>

      </main>

      <SystemMetricsBar />

    </div>
  );
}