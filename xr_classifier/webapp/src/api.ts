import axios from "axios";

export const API_BASE = "http://127.0.0.1:8000";

export const uploadXray = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await axios.post(`${API_BASE}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data;
};
