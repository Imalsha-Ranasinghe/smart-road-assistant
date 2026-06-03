// api.js — single place that talks to the Flask backend.
// CORS is enabled on the backend (flask_cors), so direct calls work.
// Override the host with a VITE_API_BASE env var if the backend runs elsewhere.
export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5000';

// Run the two-stage pipeline on one frame.
// `source` may be a File (image upload) or a Blob (canvas snapshot).
// `model` picks the detector: 'default' (photo) or 'dashcam' (video, w/ fallback).
// Returns: { annotated_image, warnings, potholes[], traffic_lights[], counts }
export async function detectImage(source, filename, model) {
  const form = new FormData();
  if (filename) form.append('image', source, filename);
  else form.append('image', source);
  if (model) form.append('model', model);

  const res = await fetch(`${API_BASE}/detect/image`, { method: 'POST', body: form });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).error || detail; } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}
