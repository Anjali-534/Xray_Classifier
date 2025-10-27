import React, { useState } from "react";
import { Container, Typography, Box } from "@mui/material";
import UploadPanel from "./components/UploadPanel";
import ResultsPanel, { Result } from "./components/ResultsPanel";

const App: React.FC = () => {
  const [result, setResult] = useState<Result | null>(null);

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 5, textAlign: "center" }}>
        <Typography variant="h4" gutterBottom>
          🩺 Chest X-Ray Analyzer
        </Typography>
      </Box>
      <UploadPanel onResult={setResult} />
      <ResultsPanel result={result} />
    </Container>
  );
};

export default App;
