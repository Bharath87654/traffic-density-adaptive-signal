import type { Lane, AnalyticsData, SystemMetrics } from '../types/traffic';

const MOCK_LANES: Lane[] = [
  {
    id: '1',
    name: 'LANE 1',
    signalStatus: 'GREEN',
    vehicles: 14,
    queue: 2,
    emergency: 'NONE',
    isActive: true,
    videoUrl: '/temp_cam_0.mp4'
  },
  {
    id: '2',
    name: 'LANE 2',
    signalStatus: 'RED',
    vehicles: 0,
    queue: 0,
    emergency: 'NONE',
    isActive: false,
    videoUrl: 'https://images.unsplash.com/photo-1545147986-a9d6f210df77?q=80&w=1200&auto=format&fit=crop'
  },
  {
    id: '3',
    name: 'LANE 3',
    signalStatus: 'RED',
    vehicles: 0,
    queue: 0,
    emergency: 'NONE',
    isActive: false,
    videoUrl: 'https://images.unsplash.com/photo-1545147986-a9d6f210df77?q=80&w=1200&auto=format&fit=crop'
  },
  {
    id: '4',
    name: 'LANE 4',
    signalStatus: 'RED',
    vehicles: 0,
    queue: 0,
    emergency: 'NONE',
    isActive: false,
    videoUrl: 'https://images.unsplash.com/photo-1545147986-a9d6f210df77?q=80&w=1200&auto=format&fit=crop'
  }
];

const MOCK_ANALYTICS: AnalyticsData = {
  trafficVolume24h: [
    { time: '0', count: 40 }, { time: '4', count: 30 }, { time: '8', count: 120 },
    { time: '12', count: 150 }, { time: '16', count: 140 }, { time: '20', count: 80 },
    { time: '24', count: 40 }
  ],
  avgWaitTimeHourly: [
    { hour: '08:00', seconds: 45 }, { hour: '09:00', seconds: 60 }, { hour: '10:00', seconds: 55 },
    { hour: '11:00', seconds: 70 }, { hour: '12:00', seconds: 85 }, { hour: '13:00', seconds: 65 },
    { hour: '14:00', seconds: 50 }, { hour: '15:00', seconds: 55 }, { hour: '16:00', seconds: 75 }
  ],
  emergencyVehicleDist: [
    { type: 'Ambulance', value: 45 },
    { type: 'Fire', value: 25 },
    { type: 'Police', value: 30 }
  ],
  vehiclesPassedCycle: Array.from({ length: 20 }, (_, i) => ({
    cycle: i + 1,
    count: Math.floor(Math.random() * 100) + 50
  }))
};

const MOCK_METRICS: SystemMetrics = {
  totalDetectedToday: 14302,
  avgWaitTime: '42s',
  aiSwitches: 115,
  peakHour: '17:00'
};

export const trafficApi = {
  fetchLanes: async (): Promise<Lane[]> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    return MOCK_LANES;
  },
  fetchAnalytics: async (): Promise<AnalyticsData> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    return MOCK_ANALYTICS;
  },
  fetchSystemMetrics: async (): Promise<SystemMetrics> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    return MOCK_METRICS;
  }
};
