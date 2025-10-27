import React from "react";
import { Card, CardMedia, CardContent, Typography } from "@mui/material";

const GradCamCard: React.FC<{ imageUrl: string }> = ({ imageUrl }) => (
  <Card sx={{ mt: 3 }}>
    <CardContent>
      <Typography variant="subtitle1" gutterBottom>🧠 Grad-CAM Visualization</Typography>
    </CardContent>
    <CardMedia
      component="img"
      height="400"
      image={imageUrl}
      alt="Grad-CAM"
      sx={{ objectFit: "contain" }}
    />
  </Card>
);

export default GradCamCard;
