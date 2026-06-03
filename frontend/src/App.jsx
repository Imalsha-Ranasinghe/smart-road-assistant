import { useMemo, useState, useRef, useCallback } from 'react';
import './App.css';
import { API_BASE, detectImage } from './api';
import Dashboard from './Dashboard';

const SEV_RANK = { High: 3, Medium: 2, Low: 1 };

// ─── Pipeline stage definitions ───────────────────────────────────────────────
const STAGES = [
  {
    key: 'input',
    title: 'Stage 1 — Input',
    description: 'Read and validate the uploaded image or video file, then extract a single frame.',
  },
  {
    key: 'preprocess',
    title: 'Stage 2 — Common Preprocessing',
    description: 'Resize to 640×640, apply CLAHE contrast, Gaussian blur, and weather-specific correction.',
  },
  {
    key: 'gate',
    title: 'Stage 3 — Gate Model',
    description: 'Run YOLOv8n to detect whether traffic lights, potholes, or both are present in the frame.',
  },
  {
    key: 'traffic',
    title: 'Stage 4A — Traffic Light Pipeline',
    description: 'Crop the region, convert to HSV, apply color thresholding, and classify Red / Yellow / Green.',
  },
  {
    key: 'pothole',
    title: 'Stage 4B — Pothole Pipeline',
    description: 'Crop the road region, apply Sobel edge detection, and classify severity with distance estimate.',
  },
  {
    key: 'postprocess',
    title: 'Stage 5 — Post Processing',
    description: 'Apply confidence threshold (≥40%), NMS to remove duplicates, and spatial filtering.',
  },
  {
    key: 'warning',
    title: 'Stage 6 — Warning Generation',
    description: 'Prioritize by type and proximity, generate visual and text alerts, trigger audio.',
  },
  {
    key: 'output',
    title: 'Stage 7 — Output',
    description: 'Draw bounding boxes and labels on the original frame and deliver the annotated result.',
  },
];

const EMPTY_SUMMARY = {
  status:       'Ready',
  trafficLight: '—',
  pothole:      '—',
  detections:   '—',
  warning:      'No active warnings',
};

// Pick the most important traffic light (lane-relevant first) and pothole (by severity)
function summarize(data) {
  const tls = data.traffic_lights ?? [];
  const phs = data.potholes ?? [];
  const primaryTl = tls.find(t => t.lane_relevant) ?? tls[0] ?? null;
  const primaryPh = [...phs].sort(
    (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0),
  )[0] ?? null;
  return { tls, phs, primaryTl, primaryPh };
}

// ─── Warning banner helper ────────────────────────────────────────────────────
// Mirrors the backend's generate_warnings(): lane-relevant traffic lights + all
// potholes, but colour-coded by the model's actual outputs.
const TL_COLOR_CLASS = { Red: 'red', Yellow: 'yellow', Green: 'green', Unknown: 'neutral' };
const TL_ICON        = { Red: '🔴', Yellow: '🟡', Green: '🟢', Unknown: '🚦' };

function WarningBanner({ result }) {
  if (!result) {
    return (
      <div className="warning-banner neutral">
        <span className="warning-icon">ℹ️</span>
        <span>No detections yet. Upload an image and run the pipeline.</span>
      </div>
    );
  }

  const items = [];

  for (const t of (result.traffic_lights ?? []).filter(t => t.lane_relevant)) {
    items.push({
      colorClass: TL_COLOR_CLASS[t.color] ?? 'neutral',
      icon:       TL_ICON[t.color] ?? '🚦',
      msg:        t.instructions,
    });
  }

  for (const p of result.potholes ?? []) {
    items.push({
      colorClass: p.severity === 'High' ? 'red' : p.severity === 'Medium' ? 'orange' : 'yellow',
      icon:       '⚠️',
      msg:        p.instructions,
    });
  }

  if (items.length === 0) {
    return (
      <div className="warning-banner green">
        <span className="warning-icon">✅</span>
        <span>Road clear — No hazards detected in this frame.</span>
      </div>
    );
  }

  return (
    <>
      {items.map((item, i) => (
        <div key={i} className={`warning-banner ${item.colorClass}`}>
          <span className="warning-icon">{item.icon}</span>
          <span>{item.msg}</span>
        </div>
      ))}
    </>
  );
}

// ─── Detection Pipeline page (single-image, full stage trace) ─────────────────
function PipelinePage() {
  const [activeStages, setActiveStages]   = useState([]);
  const [running, setRunning]             = useState(false);
  const [summary, setSummary]             = useState(EMPTY_SUMMARY);
  const [result, setResult]               = useState(null);
  const [error, setError]                 = useState(null);
  const [history, setHistory]             = useState([]);
  const [imageFile, setImageFile]         = useState(null);
  const [imagePreview, setImagePreview]   = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const fileInputRef = useRef(null);

  // Handle file selection
  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setActiveStages([]);
    setSummary(EMPTY_SUMMARY);
  }, []);

  const onFileChange = (e) => handleFile(e.target.files[0]);
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // Run pipeline — calls the real Flask backend (trained detector + severity models)
  const runPipeline = async () => {
    if (!imageFile) return;
    setRunning(true);
    setResult(null);
    setError(null);
    setActiveStages(['input', 'preprocess', 'gate']);
    setSummary({ status: 'Processing…', trafficLight: 'Scanning', pothole: 'Scanning', detections: '…', warning: 'Analyzing frame…' });

    try {
      const data = await detectImage(imageFile);
      const { tls, phs, primaryTl, primaryPh } = summarize(data);

      // Reflect which branches actually ran in the stage tracker
      const stages = ['input', 'preprocess', 'gate'];
      if (tls.length) stages.push('traffic');
      if (phs.length) stages.push('pothole');
      stages.push('postprocess', 'warning', 'output');
      setActiveStages(stages);

      const tlLabel = primaryTl
        ? `${primaryTl.color}${primaryTl.lane_relevant ? '' : ' (adjacent)'}${tls.length > 1 ? ` +${tls.length - 1}` : ''}`
        : 'Not detected';
      const phLabel = primaryPh
        ? `${primaryPh.severity} — ${primaryPh.distance}${phs.length > 1 ? ` +${phs.length - 1}` : ''}`
        : 'Not detected';

      setSummary({
        status:       'Completed',
        trafficLight: tlLabel,
        pothole:      phLabel,
        detections:   `${data.counts.potholes} PH · ${data.counts.traffic_lights} TL`,
        warning:      (data.warnings?.length) ? 'Active warnings — see below' : 'No hazards detected',
      });

      setResult(data);
      setHistory(prev => [{
        id:      Date.now(),
        time:    new Date().toLocaleTimeString(),
        traffic: primaryTl ? primaryTl.color : '—',
        pothole: primaryPh ? primaryPh.severity : '—',
        counts:  `${data.counts.potholes + data.counts.traffic_lights} obj`,
      }, ...prev].slice(0, 6));
    } catch (e) {
      setError(e.message || String(e));
      setActiveStages([]);
      setSummary({ status: 'Error', trafficLight: '—', pothole: '—', detections: '—', warning: 'Backend request failed' });
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setActiveStages([]);
    setRunning(false);
    setSummary(EMPTY_SUMMARY);
    setResult(null);
    setError(null);
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const activeCount = useMemo(() => activeStages.length, [activeStages]);

  return (
    <div className="app-shell">

      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <p className="eyebrow">Smart Road Assistant</p>
          <h1>Detection Pipeline</h1>
          <p className="subtitle">
            Upload a road image, trace every stage — from preprocessing through the gate model
            to traffic light and pothole detection — and get a live annotated result.
          </p>
        </div>
        <div className="controls">
          <button
            className="primary-button"
            onClick={runPipeline}
            disabled={running || !imageFile}
          >
            {running ? 'Running…' : 'Run Pipeline'}
          </button>
          <button className="secondary-button" onClick={reset}>Reset</button>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="content-grid">

        {/* ── Left column: Upload + Stages ── */}
        <section className="panel">

          {/* Upload zone */}
          <div className="panel-header">
            <h2>Input image</h2>
            {imageFile && (
              <span>{imageFile.name.length > 28 ? imageFile.name.slice(0,28)+'…' : imageFile.name}</span>
            )}
          </div>

          {!imagePreview ? (
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={onFileChange}
              />
              <span className="upload-icon">🛣️</span>
              <strong>Drop a road image here</strong>
              <p>or click to browse — JPG, PNG, WEBP supported</p>
            </div>
          ) : (
            <div style={{ marginBottom: '20px' }}>
              <span className="preview-label">Input frame</span>
              <img src={imagePreview} alt="Input" className="preview-img" />
              <button
                className="secondary-button"
                onClick={() => { setImageFile(null); setImagePreview(null); setResult(null); setActiveStages([]); setSummary(EMPTY_SUMMARY); }}
                style={{ fontSize: '0.8rem', padding: '7px 14px' }}
              >
                ✕ Remove image
              </button>
            </div>
          )}

          {/* Processing bar */}
          {running && (
            <div className="processing-bar" style={{ marginBottom: '16px' }}>
              <div className="processing-bar-fill" />
            </div>
          )}

          {/* Pipeline stages */}
          <div className="panel-header" style={{ marginTop: '4px' }}>
            <h2>Pipeline stages</h2>
            <span>{activeCount} / {STAGES.length} active</span>
          </div>

          <div className="stage-list">
            {STAGES.map(stage => {
              const active = activeStages.includes(stage.key);
              return (
                <article key={stage.key} className={`stage-card ${active ? 'active' : ''}`}>
                  <div className="stage-title">
                    <strong>{stage.title}</strong>
                    <span className="stage-badge">{active ? 'Active' : 'Pending'}</span>
                  </div>
                  <p>{stage.description}</p>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── Right column: Summary + Result ── */}
        <aside className="panel">

          {/* Run summary */}
          <div className="panel-header">
            <h2>Run summary</h2>
            {summary.status === 'Completed' && <span>✓ Done</span>}
          </div>

          <div className="stats-grid">
            {[
              { label: 'Status',        value: summary.status       },
              { label: 'Detections',    value: summary.detections   },
              { label: 'Traffic Light', value: summary.trafficLight },
              { label: 'Pothole',       value: summary.pothole      },
            ].map(({ label, value }) => (
              <div className="stat-card" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          {/* Annotated result image — boxes + labels drawn by the backend pipeline */}
          {result?.annotated_image && summary.status === 'Completed' && (
            <div className="result-section">
              <span className="preview-label">Annotated output</span>
              <img
                src={`data:image/jpeg;base64,${result.annotated_image}`}
                alt="Annotated result"
                className="preview-img"
              />
            </div>
          )}

          {/* Warnings */}
          <div style={{ marginTop: '16px', marginBottom: '20px' }}>
            {error ? (
              <div className="warning-banner red">
                <span className="warning-icon">⛔</span>
                <span>{error} — is the backend running at {API_BASE}? Start it with: python backend/app.py</span>
              </div>
            ) : (
              <WarningBanner result={result} />
            )}
          </div>

          {/* History */}
          <div className="history-panel">
            <h3>Recent runs</h3>
            {history.length === 0 ? (
              <p className="empty-state">No runs yet.</p>
            ) : (
              <ul>
                {history.map(run => (
                  <li key={run.id} className="history-item">
                    <span className="history-text">
                      TL: <strong>{run.traffic}</strong> &nbsp;·&nbsp; PH: <strong>{run.pothole}</strong> &nbsp;·&nbsp; {run.counts}
                    </span>
                    <span className="history-time">{run.time}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

        </aside>
      </main>
    </div>
  );
}

// ─── Root: nav shell switching between Dashboard and Pipeline ──────────────────
export default function App() {
  const [page, setPage] = useState('dashboard');

  return (
    <div className="app-root">
      <nav className="app-nav">
        <span className="app-nav-brand">🛣️ Smart Road Assistant</span>
        <div className="app-nav-tabs">
          <button
            className={page === 'dashboard' ? 'active' : ''}
            onClick={() => setPage('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={page === 'pipeline' ? 'active' : ''}
            onClick={() => setPage('pipeline')}
          >
            Detection Pipeline
          </button>
        </div>
      </nav>

      {page === 'dashboard' ? <Dashboard /> : <PipelinePage />}
    </div>
  );
}