import React, { useState } from "react";
import { Button, LinearProgress, Box } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import axios from "axios";
import type { Result } from "./ResultsPanel";

interface Props { onResult: (r: Result) => void; }

const UploadPanel: React.FC<Props> = ({ onResult }) => {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();  
    formData.append("file", file);
  const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  body: formData,
})


    setLoading(true);
    try {
      const res = await axios.post<Result>(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (p) => setProgress((p.loaded / (p.total ?? 1)) * 100)
        }
      );
      onResult(res.data);
    } catch (err) {
      alert("Upload failed. Check server.");
      console.error(err);
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <Box textAlign="center" sx={{ mb: 4 }}>
      <input
        accept="image/*"
        style={{ display: "none" }}
        id="upload-btn"
        type="file"
        onChange={handleUpload}
      />
      <label htmlFor="upload-btn">
        <Button
          variant="contained"
          component="span"
          startIcon={<CloudUploadIcon />}
          disabled={loading}
        >
          Upload X-Ray Image
        </Button>
      </label>
      {loading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress variant="determinate" value={progress} />
        </Box>
      )}
    </Box>
  );
};

export default UploadPanel;
