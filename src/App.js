import React, { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const videoRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) {
      setMessage("Please select a video file.");
      return;
    }

    if (!file.type.startsWith('video/')) {
      setMessage("Please upload a valid video file.");
      return;
    }

    setVideoFile(file);
    setMessage("");
    setResult(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    if (videoRef.current) {
      videoRef.current.src = url;
    }
  };

  const analyzeVideo = async () => {
    if (!videoFile) {
      setMessage("Please select a video first!");
      return;
    }

    setMessage("Processing video...");
    setIsLoading(true);
    setResult(null);
    setProgress(0);

    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/analyze_video",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: progressEvent => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          }
        }
      );

      if (response.data.status === "success") {
        setResult(response.data);
        setMessage(response.data.message);
      } else {
        setMessage(response.data.message || "Analysis failed");
      }
    } catch (error) {
      const errorMessage =
        error.response?.data?.message ||
        error.message ||
        "Unknown error occurred";
      setMessage(`Error: ${errorMessage}`);
    } finally {
      setIsLoading(false);
      setProgress(0);
    }
  };

  return (
    <div className="container">
      <h1>Deepfake Detection System</h1>

      <div className="upload-section">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          disabled={isLoading}
        />
        <button
          onClick={analyzeVideo}
          disabled={isLoading || !videoFile}
        >
          {isLoading ? `Processing... ${progress}%` : "Analyze Video"}
        </button>
      </div>

      {message && (
        <div
          className={`message ${
            message.startsWith("Error") ? "error" : "info"
          }`}
        >
          {message}
        </div>
      )}

      <div className="preview-section">
        {videoFile && (
          <video
            ref={videoRef}
            controls
            style={{ maxWidth: "100%", marginTop: "20px" }}
          />
        )}
      </div>

      {result && (
        <div className="result-section">
          <h2>Analysis Result:</h2>
          <div className={`result ${result.result?.toLowerCase()}`}>
            <p>Status: {result.result}</p>
            <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
