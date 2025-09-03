import { Container, Typography, Box, Paper, LinearProgress, Snackbar, Alert } from '@mui/material'
import { useState } from 'react'
import UploadCard from './components/UploadCard'
import ResultsPanel, { ApiResult } from './components/ResultsPanel'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ApiResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onFileSelected = async (file: File) => {
    setError(null); setResult(null); setLoading(true)
    const form = new FormData()
    form.append('file', file)
    try {
      const resp = await axios.post(`${API_BASE}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(resp.data)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight={700}>X‑ray Classifier (Demo)</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Research demo — not for clinical use.
      </Typography>

      <Box sx={{ my: 2 }}>
        <UploadCard onFile={onFileSelected} />
      </Box>

      {loading && (
        <Paper sx={{ p: 2, my: 2 }}>
          <Typography>Running inference…</Typography>
          <LinearProgress sx={{ mt: 1 }} />
        </Paper>
      )}

      {result && <ResultsPanel result={result} />}

      <Snackbar open={!!error} autoHideDuration={4000} onClose={() => setError(null)}>
        <Alert severity="error" variant="filled">{error}</Alert>
      </Snackbar>
    </Container>
  )
}

export default App
