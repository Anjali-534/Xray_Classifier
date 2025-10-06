import React, { useState } from "react";
import { uploadImage } from "../api";

interface Props {
  onResult: (data: unknown) => void;
}

export const UploadPanel: React.FC<Props> = ({ onResult }) => {
  const [file, setFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (file) {
      const result = await uploadImage(file);
      onResult(result);
    }
  };

  return (
    <div className="p-3 border rounded">
      <h5>Upload Chest X-Ray</h5>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload} className="btn btn-primary mt-2">
        Analyze
      </button>
    </div>
  );
};
