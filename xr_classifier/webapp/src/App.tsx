import React, { useState } from "react";
import { UploadPanel } from "./components/UploadPanel";
import { ResultsPanel } from "./components/ResultsPanel";

import type { Result } from "./components/ResultsPanel";

const App: React.FC = () => {
  const [result, setResult] = useState<Result | null>(null);

  return (
    <div className="container mt-5">
      <h2 className="mb-4 text-center">🩺 Chest X-Ray Analyzer</h2>
      <UploadPanel onResult={setResult} />
      <ResultsPanel result={result} />
    </div>
  );
};

export default App;
