import { Paper, Typography, Box, Slider, Grid, LinearProgress } from '@mui/material'
import { useMemo, useState } from 'react'

export type ApiResult = {
  classes: string[]
  probs: number[]
  probs_cal?: number[]
  uncertainty_std?: number[]
  topk: { class: string; p: number }[]
  overlay_png_base64: string
  model_version: string
  disclaimer: string
}

export default function ResultsPanel({ result }: { result: ApiResult }) {
  const [alpha, setAlpha] = useState<number>(70)

  const overlayUrl = useMemo(
    () => `data:image/png;base64,${result.overlay_png_base64}`,
    [result.overlay_png_base64]
  )

  return (
    <Paper sx={{ p: 2, my: 2 }}>
      <Typography variant="h6" gutterBottom>Results</Typography>
      <Typography variant="body2" color="text.secondary">
        Model: {result.model_version} — {result.disclaimer}
      </Typography>

      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ position: 'relative', width: '100%', borderRadius: 1, overflow: 'hidden' }}>
            {/* For stub we only show overlay heatmap */}
            <img src={overlayUrl} style={{ width: '100%', opacity: alpha/100 }} />
          </Box>
          <Box sx={{ mt: 1 }}>
            <Typography gutterBottom>Heatmap Opacity</Typography>
            <Slider value={alpha} onChange={(_, v) => setAlpha(v as number)} />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          {result.classes.map((c, i) => {
            const p = result.probs[i]
            return (
              <Box key={c} sx={{ mb: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{c} — {(p*100).toFixed(0)}%</Typography>
                <LinearProgress variant="determinate" value={p*100} />
              </Box>
            )
          })}
        </Grid>
      </Grid>
    </Paper>
  )
}
