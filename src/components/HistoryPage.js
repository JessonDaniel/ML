import React, { useState, useEffect } from 'react';
import axios from 'axios';

function HistoryPage() {
  const [videos, setVideos] = useState([]);

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await axios.get('http://localhost:8000/videos');
      setVideos(response.data.videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  return (
    <div className="history-container">
      {videos.map((video) => (
        <div key={video._id} className="history-card">
          <h3>{video.filename}</h3>
          <div className="analysis-details">
            {typeof video.analysis === 'object' ? (
              <>
                <p>Duration: {video.analysis.video_stats?.duration}s</p>
                <p>Detections: {video.analysis.detection_summary?.total_detections}</p>
                <p>Classes: {video.analysis.detection_summary?.unique_classes.join(', ')}</p>
              </>
            ) : (
              <p>{video.analysis}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default HistoryPage;
