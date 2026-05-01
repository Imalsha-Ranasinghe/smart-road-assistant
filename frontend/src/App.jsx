import { useMemo, useState } from 'react';
import './App.css';

const pipelineStages = [
  {
    title: 'Stage 1 — Input',
    description: 'Read image/video/camera input, validate it, and extract a single frame for processing.',
  },
  {
    title: 'Stage 2 — Common Preprocessing',
    description: 'Resize, enhance contrast, blur noise, and apply weather-specific correction before analysis.',
  },
  {
    title: 'Stage 3 — Initial Gate Model',
    description: 'Run the lightweight gate model to detect traffic lights, potholes, or both.',
  },
  {
    title: 'Stage 4A — Traffic Light Pipeline',
    description: 'Crop traffic light region, classify color, and run the traffic light model for final output.',
  },
  {
    title: 'Stage 4B — Pothole Pipeline',
    description: 'Crop road region, enhance edges, and run the pothole model to determine severity and distance.',
  },
  {
    title: 'Stage 5 — Post Processing',
    description: 'Filter low-confidence and overlapping detections, and clean final outputs from both pipelines.',
  },
  {
    title: 'Stage 6 — Warning Generation',
    description: 'Prioritize warnings, generate visual/text alerts, and trigger audio notifications.',
  },
  {
    title: 'Stage 7 — Output',
    description: 'Draw annotations, send the frame to the web UI, and provide an alert-ready annotated image.',
  },
];

const initialSummary = {
  status: 'Ready to run',
  trafficLight: 'Unknown',
  pothole: 'Unknown',
  warning: 'No active warnings',
};

function App() {
  const [summary, setSummary] = useState(initialSummary);
  const [activeStages, setActiveStages] = useState([]);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);

  const runPipeline = () => {
    setRunning(true);
    setActiveStages(['Stage 1 — Input']);
    setSummary({ status: 'Processing...', trafficLight: 'Scanning', pothole: 'Scanning', warning: 'Analyzing frame' });

    setTimeout(() => {
      setActiveStages(['Stage 1 — Input', 'Stage 2 — Common Preprocessing', 'Stage 3 — Initial Gate Model']);
    }, 600);

    setTimeout(() => {
      const trafficFound = Math.random() > 0.35;
      const potholeFound = Math.random() > 0.35;
      const tlState = trafficFound ? ['Red', 'Yellow', 'Green'][Math.floor(Math.random() * 3)] : 'None';
      const phSeverity = potholeFound ? ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)] : 'None';
      const warningLabel = trafficFound
        ? `Traffic light detected: ${tlState}`
        : potholeFound
        ? `Pothole detected: ${phSeverity}`
        : 'No warnings detected';

      setActiveStages([
        'Stage 1 — Input',
        'Stage 2 — Common Preprocessing',
        'Stage 3 — Initial Gate Model',
        trafficFound && 'Stage 4A — Traffic Light Pipeline',
        potholeFound && 'Stage 4B — Pothole Pipeline',
        'Stage 5 — Post Processing',
        'Stage 6 — Warning Generation',
        'Stage 7 — Output',
      ].filter(Boolean));

      setSummary({
        status: 'Completed',
        trafficLight: trafficFound ? `Detected (${tlState})` : 'Not detected',
        pothole: potholeFound ? `Detected (${phSeverity})` : 'Not detected',
        warning: warningLabel,
      });

      setHistory(prev => [
        {
          id: prev.length + 1,
          result: warningLabel,
          time: new Date().toLocaleTimeString(),
          trafficLight: tlState,
          pothole: phSeverity,
        },
        ...prev,
      ]);
      setRunning(false);
    }, 1600);
  };

  const resetPipeline = () => {
    setSummary(initialSummary);
    setActiveStages([]);
    setRunning(false);
  };

  const completedStages = useMemo(() => activeStages.length, [activeStages]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Smart Road Assistant</p>
          <h1>Visual pipeline dashboard</h1>
          <p className="subtitle">Monitor each stage from input capture through traffic-light and pothole warnings.</p>
        </div>
        <div className="controls">
          <button className="primary-button" onClick={runPipeline} disabled={running}>
            {running ? 'Running...' : 'Run Pipeline'}
          </button>
          <button className="secondary-button" onClick={resetPipeline}>
            Reset
          </button>
        </div>
      </header>

      <main className="content-grid">
        <section className="panel panel-large">
          <div className="panel-header">
            <h2>Pipeline stages</h2>
            <span>{completedStages} / {pipelineStages.length} active</span>
          </div>

          <div className="stage-list">
            {pipelineStages.map(stage => {
              const active = activeStages.includes(stage.title);
              return (
                <article key={stage.title} className={`stage-card ${active ? 'active' : ''}`}>
                  <div className="stage-title">
                    <strong>{stage.title}</strong>
                    <span>{active ? 'Active' : 'Pending'}</span>
                  </div>
                  <p>{stage.description}</p>
                </article>
              );
            })}
          </div>
        </section>

        <aside className="panel panel-side">
          <div className="panel-header">
            <h2>Run summary</h2>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <span>Status</span>
              <strong>{summary.status}</strong>
            </div>
            <div className="stat-card">
              <span>Traffic Light</span>
              <strong>{summary.trafficLight}</strong>
            </div>
            <div className="stat-card">
              <span>Pothole</span>
              <strong>{summary.pothole}</strong>
            </div>
            <div className="stat-card">
              <span>Warning</span>
              <strong>{summary.warning}</strong>
            </div>
          </div>

          <div className="history-panel">
            <h3>Recent pipeline runs</h3>
            {history.length === 0 ? (
              <p>No pipeline runs yet.</p>
            ) : (
              <ul>
                {history.slice(0, 4).map(run => (
                  <li key={run.id}>
                    <strong>{run.time}</strong>
                    <p>{run.result}</p>
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

export default App;
