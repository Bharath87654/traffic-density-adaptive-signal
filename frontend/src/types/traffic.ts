export interface Lane {
  id: string;
  name: string;
  signalStatus: 'GREEN' | 'RED';
  vehicles: number;
  queue: number;
  emergency: 'NONE' | 'AMBULANCE' | 'FIRE' | 'POLICE';
  videoUrl?: string;
  isActive: boolean;
}

export interface AnalyticsData {
  trafficVolume24h: { time: string; count: number }[];
  avgWaitTimeHourly: { hour: string; seconds: number }[];
  emergencyVehicleDist: { type: string; value: number }[];
  vehiclesPassedCycle: { cycle: number; count: number }[];
}

export interface SystemMetrics {
  totalDetectedToday: number;
  avgWaitTime: string;
  aiSwitches: number;
  peakHour: string;
}

export interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
}
