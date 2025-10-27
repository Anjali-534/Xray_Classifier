import React from "react";
import { Typography, Paper, Divider, List, ListItem } from "@mui/material";
import GradCamCard from "./GradCamCard";

export interface Result {
  predicted_class: string;
  confidence: number;
  uncertainty: number;
  gradcam_url: string;
}

interface Props { result: Result | null; }

const ResultsPanel: React.FC<Props> = ({ result }) => {
  if (!result) return null;

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h6" gutterBottom>📋 Prediction Results</Typography>
      <Divider sx={{ mb: 2 }} />
      <List>
        <ListItem>🧩 Class: <strong>{result.predicted_class}</strong></ListItem>
        <ListItem>🔢 Confidence: {(result.confidence * 100).toFixed(2)}%</ListItem>
        <ListItem>📉 Uncertainty: {(result.uncertainty * 100).toFixed(2)}%</ListItem>
      </List>
      {result.gradcam_url && <GradCamCard imageUrl={result.gradcam_url} />}
    </Paper>
  );
};

export default ResultsPanel;
