import { useMemo, useState, useRef, useCallback } from 'react';
import './App.css';

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
  weather:      '—',
  warning:      'No active warnings',
};

// ─── Simulate pipeline phases ─────────────────────────────────────────────────
function simulatePipeline(hasImage, onStageUpdate, onComplete) {
  const trafficFound = Math.random() > 0.35;
  const potholeFound = Math.random() > 0.35;
  const tlState      = trafficFound ? ['Red', 'Yellow', 'Green'][Math.floor(Math.random() * 3)] : null;
  const phSeverity   = potholeFound ? ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)]  : null;
  const phDistance   = potholeFound ? ['Very Close', 'Close', 'Medium', 'Far'][Math.floor(Math.random() * 4)] : null;
  const weathers     = ['Clear', 'Foggy', 'Rainy', 'Night'];
  const weather      = weathers[Math.floor(Math.random() * weathers.length)];

  const phases = [
    { stages: ['input'],                               delay: 300  },
    { stages: ['input','preprocess'],                  delay: 600  },
    { stages: ['input','preprocess','gate'],            delay: 1000 },
    {
      stages: [
        'input','preprocess','gate',
        trafficFound ? 'traffic' : null,
        potholeFound ? 'pothole' : null,
      ].filter(Boolean),
      delay: 1400,
    },
    {
      stages: [
        'input','preprocess','gate',
        trafficFound ? 'traffic' : null,
        potholeFound ? 'pothole' : null,
        'postprocess','warning','output',
      ].filter(Boolean),
      delay: 1900,
    },
  ];

  phases.forEach(({ stages, delay }) => {
    setTimeout(() => onStageUpdate(stages), delay);
  });

  setTimeout(() => {
    onComplete({ trafficFound, potholeFound, tlState, phSeverity, phDistance, weather });
  }, 1900);
}

// ─── Warning banner helper ────────────────────────────────────────────────────
function WarningBanner({ result }) {
  if (!result) {
    return (
      <div className="warning-banner neutral">
        <span className="warning-icon">ℹ️</span>
        <span>No detections yet. Upload an image and run the pipeline.</span>
      </div>
    );
  }

  const { trafficFound, potholeFound, tlState, phSeverity, phDistance } = result;
  const items = [];

  if (trafficFound) {
    const colorClass = tlState === 'Red' ? 'red' : tlState === 'Yellow' ? 'yellow' : 'green';
    const icon       = tlState === 'Red' ? '🔴' : tlState === 'Yellow' ? '🟡' : '🟢';
    const msg        = tlState === 'Red'
      ? 'STOP — Red traffic light detected!'
      : tlState === 'Yellow'
      ? 'SLOW DOWN — Yellow traffic light ahead.'
      : 'Green light — Safe to proceed.';
    items.push({ colorClass, icon, msg });
  }

  if (potholeFound) {
    const sev   = phSeverity;
    const clr   = sev === 'High' ? 'red' : sev === 'Medium' ? 'orange' : 'yellow';
    const icon  = '⚠️';
    const msg   = `POTHOLE AHEAD — ${sev} severity, ${phDistance}.`;
    items.push({ colorClass: clr, icon, msg });
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

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeStages, setActiveStages]   = useState([]);
  const [running, setRunning]             = useState(false);
  const [summary, setSummary]             = useState(EMPTY_SUMMARY);
  const [result, setResult]               = useState(null);
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
    setActiveStages([]);
    setSummary(EMPTY_SUMMARY);
  }, []);

  const onFileChange = (e) => handleFile(e.target.files[0]);
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // Run pipeline
  const runPipeline = () => {
    if (!imageFile) return;
    setRunning(true);
    setResult(null);
    setSummary({ status: 'Processing…', trafficLight: 'Scanning', pothole: 'Scanning', weather: 'Detecting', warning: 'Analyzing frame…' });

    simulatePipeline(
      true,
      (stages) => setActiveStages(stages),
      ({ trafficFound, potholeFound, tlState, phSeverity, phDistance, weather }) => {
        const tlLabel = trafficFound ? `Detected — ${tlState}` : 'Not detected';
        const phLabel = potholeFound ? `Detected — ${phSeverity} severity, ${phDistance}` : 'Not detected';

        setSummary({
          status:       'Completed',
          trafficLight: tlLabel,
          pothole:      phLabel,
          weather:      weather,
          warning:      trafficFound || potholeFound ? 'Active warnings — see below' : 'No hazards detected',
        });

        const r = { trafficFound, potholeFound, tlState, phSeverity, phDistance };
        setResult(r);
        setHistory(prev => [{
          id:       Date.now(),
          time:     new Date().toLocaleTimeString(),
          traffic:  trafficFound ? tlState : '—',
          pothole:  potholeFound ? phSeverity : '—',
          weather,
        }, ...prev].slice(0, 6));
        setRunning(false);
      }
    );
  };

  const reset = () => {
    setActiveStages([]);
    setRunning(false);
    setSummary(EMPTY_SUMMARY);
    setResult(null);
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
              { label: 'Weather',       value: summary.weather      },
              { label: 'Traffic Light', value: summary.trafficLight },
              { label: 'Pothole',       value: summary.pothole      },
            ].map(({ label, value }) => (
              <div className="stat-card" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          {/* Annotated result image (reuses input until backend provides one) */}
          {imagePreview && summary.status === 'Completed' && (
            <div className="result-section">
              <span className="preview-label">Annotated output</span>
              <img src={imagePreview} alt="Annotated result" className="preview-img" />
            </div>
          )}

          {/* Warnings */}
          <div style={{ marginTop: '16px', marginBottom: '20px' }}>
            <WarningBanner result={result} />
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
                      TL: <strong>{run.traffic}</strong> &nbsp;·&nbsp; PH: <strong>{run.pothole}</strong> &nbsp;·&nbsp; {run.weather}
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