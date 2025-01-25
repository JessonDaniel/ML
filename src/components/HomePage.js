import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Video,
  BarChart2,
  FileText,
  Clock 
} from 'lucide-react';

function HomePage() {
  const [stats, setStats] = useState({
    totalVideos: 0,
    totalAnalyses: 0,
    recentDetections: 0,
    averageProcessingTime: 0
  });

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        const response = await axios.get('http://localhost:8000/dashboard-stats');
        setStats(response.data);
      } catch (error) {
        console.error('Error fetching dashboard stats:', error);
      }
    };

    fetchDashboardStats();
  }, []);

  return (
    <div>
      <div className="dashboard grid grid-cols-4 gap-4 mb-8">
        <div className="dashboard-card bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <Video className="text-blue-500" />
            <span className="text-2xl font-bold">{stats.totalVideos}</span>
          </div>
          <p className="text-sm text-gray-500 mt-2">Total Videos</p>
        </div>
        <div className="dashboard-card bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <BarChart2 className="text-green-500" />
            <span className="text-2xl font-bold">{stats.totalAnalyses}</span>
          </div>
          <p className="text-sm text-gray-500 mt-2">Total Analyses</p>
        </div>
        <div className="dashboard-card bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <FileText className="text-purple-500" />
            <span className="text-2xl font-bold">{stats.recentDetections}</span>
          </div>
          <p className="text-sm text-gray-500 mt-2">Recent Detections</p>
        </div>
        <div className="dashboard-card bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <Clock className="text-orange-500" />
            <span className="text-2xl font-bold">
              {stats.averageProcessingTime}s
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-2">Avg Processing</p>
        </div>
      </div>
      {/* Optional: Add welcome content or recent activity */}
      <div className="welcome-section bg-white p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-semibold mb-4">Welcome to Video Analysis System</h2>
        <p className="text-gray-600">Here you can upload, analyze, and explore your videos.</p>
      </div>
    </div>
  );
}

export default HomePage