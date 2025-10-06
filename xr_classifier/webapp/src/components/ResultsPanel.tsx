// src/components/ResultsPanel.tsx
import React from 'react';

export interface Result {
  predicted_class: string;
  confidence: number;
  gradcam_url?: string;
}

export const ResultsPanel: React.FC<{ result: Result | null }> = ({ result }) => {
  if (!result) return null;

  return (
    <div className="mt-4">
      <h5>Prediction Results</h5>
      <p>
        <strong>Predicted Class:</strong> {result.predicted_class}
      </p>
      <p>
        <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
      </p>
      {result.gradcam_url && (
        <div className="mt-3">
          <h6>Grad-CAM Visualization</h6>
          <img
            src={result.gradcam_url}
            alt="Grad-CAM"
            className="img-fluid rounded shadow"
          />
        </div>
      )}
    </div>
  );
};
