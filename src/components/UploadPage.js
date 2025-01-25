import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, FileVideo } from 'lucide-react';

function UploadPage() {
  const [video, setVideo] = useState(null);
  const [message, setMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  const handleVideoSubmit = async () => {
    if (!video) {
      setMessage('Please select a video to upload.');
      return;
    }

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', video);

    try {
      await axios.post('http://localhost:8000/upload-video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage('Video analysis started!');
    } catch (error) {
      setMessage('Upload failed. Try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('video/')) {
      setVideo(file);
      setMessage('');
    } else {
      setMessage('Invalid file type. Upload a video.');
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
      <div 
        className={`
          border-2 border-dashed rounded-xl p-12 text-center 
          transition-colors duration-300
          ${isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-blue-300'
          }
        `}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault(); 
          setIsDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFileSelect(e.dataTransfer.files[0]);
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        {isAnalyzing ? (
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin text-blue-500">
              <Upload className="w-12 h-12" />
            </div>
            <p className="text-gray-600">Analyzing video...</p>
          </div>
        ) : video ? (
          <div className="flex flex-col items-center space-y-4">
            <FileVideo className="w-12 h-12 text-blue-500" />
            <p className="text-gray-700">{video.name}</p>
            <button 
              onClick={handleVideoSubmit}
              className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition"
            >
              Start Analysis
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-4">
            <Upload className="w-12 h-12 text-gray-400" />
            <p className="text-gray-600">
              Drag & drop video or click to upload
            </p>
          </div>
        )}
        
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={(e) => handleFileSelect(e.target.files[0])}
          className="hidden"
        />
      </div>

      {message && (
        <div className={`
          mt-4 p-3 rounded-lg text-center
          ${message.includes('failed') 
            ? 'bg-red-100 text-red-700' 
            : 'bg-green-100 text-green-700'
          }
        `}>
          {message}
        </div>
      )}
    </div>
  );
}

export default UploadPage;
