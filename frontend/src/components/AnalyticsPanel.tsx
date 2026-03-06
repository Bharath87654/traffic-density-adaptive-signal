import React from 'react';
import type { AnalyticsData } from '../types/traffic';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { Settings } from 'lucide-react';

interface AnalyticsPanelProps {
  data: AnalyticsData;
}

const COLORS = ['#00ff88', '#ffcc00', '#ff4d4d', '#3b82f6'];

const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({ data }) => {
  return (
    <div className="dashboard-card p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-widest">AnalyticsPanel</h2>
        <Settings className="w-4 h-4 text-slate-500 cursor-pointer hover:text-white" />
      </div>

      <div className="grid grid-cols-2 gap-6 flex-1">
        {/* Traffic Volume */}
        <div className="bg-black/20 p-4 rounded-lg border border-traffic-border/30">
          <h3 className="text-slate-400 text-[10px] uppercase tracking-wider mb-2">Traffic Volume (24h)</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.trafficVolume24h}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                <XAxis dataKey="time" hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px' }}
                  itemStyle={{ color: '#00ff88' }}
                />
                <Line type="monotone" dataKey="count" stroke="#00ff88" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Avg Waiting Time */}
        <div className="bg-black/20 p-4 rounded-lg border border-traffic-border/30">
          <h3 className="text-slate-400 text-[10px] uppercase tracking-wider mb-2">Avg Waiting Time (Hourly)</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.avgWaitTimeHourly}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                <XAxis dataKey="hour" hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px' }}
                />
                <Bar dataKey="seconds" fill="#ffcc00" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Emergency Distribution */}
        <div className="bg-black/20 p-4 rounded-lg border border-traffic-border/30">
          <h3 className="text-slate-400 text-[10px] uppercase tracking-wider mb-2">Emergency Vehicle Distribution</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.emergencyVehicleDist}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={45}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {data.emergencyVehicleDist.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Vehicles Passed per Cycle */}
        <div className="bg-black/20 p-4 rounded-lg border border-traffic-border/30">
          <h3 className="text-slate-400 text-[10px] uppercase tracking-wider mb-2">Vehicles Passed per Cycle</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.vehiclesPassedCycle.slice(-10)}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                <XAxis dataKey="cycle" hide />
                <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPanel;
