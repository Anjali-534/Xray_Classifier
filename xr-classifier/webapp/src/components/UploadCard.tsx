import { Paper, Typography, Button } from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import { useRef } from 'react'

export default function UploadCard({ onFile }: { onFile: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
  }

  return (
    <Paper sx={{ p: 3, textAlign: 'center' }} variant="outlined">
      <CloudUploadIcon color="primary" sx={{ fontSize: 48 }} />
      <Typography variant="h6" sx={{ mt: 1 }}>Upload a chest X‑ray (PNG/JPG)</Typography>
      <input ref={inputRef} type="file" accept="image/png,image/jpeg" hidden onChange={onChange} />
      <Button sx={{ mt: 2 }} variant="contained" onClick={() => inputRef.current?.click()}>
        Choose image
      </Button>
    </Paper>
  )
}
