import { useRef, useState, useEffect } from 'react';
import { detectImage, API_BASE } from './api';
import * as audioWarnings from './audioWarnings';
import './Dashboard.css';

// How long to wait between analysis cycles (backend latency usually dominates).
const ANALYZE_GAP_MS = 200;
// JPEG quality for the captured frame — lower = smaller upload = faster round-trip.
const JPEG_QUALITY = 0.6;
// Only speak alerts at or above this priority score.
// Set to 30 so all pothole severities and traffic light colors are spoken.
// Unknown/unclear signals (score 20) are intentionally excluded.
const SPEAK_MIN_SCORE = 30;
// Avoid filling history with the same detection on every analyzed frame.
const HISTORY_REPEAT_GAP_MS = 2000;

// Turn a backend result into a priority-sorted list of driver alerts.
// Mirrors the backend's generate_warnings(): lane-relevant lights + all potholes.
function buildAlerts(data) {
  if (!data) return [];
  const out = [];

  for (const t of (data.traffic_lights ?? []).filter(t => t.lane_relevant)) {
    const score = t.color === 'Red' ? 100 : t.color === 'Yellow' ? 70 : t.color === 'Green' ? 40 : 20;
    out.push({
      score,
      key:        audioWarnings.trafficKey(t),
      colorClass: t.color === 'Red' ? 'red' : t.color === 'Yellow' ? 'yellow' : t.color === 'Green' ? 'green' : 'neutral',
      icon:       t.color === 'Red' ? '🛑' : t.color === 'Yellow' ? '🚦' : t.color === 'Green' ? '✅' : '🚦',
      title:      `${t.color} light ahead`,
      msg:        t.instructions,
    });
  }

  for (const p of data.potholes ?? []) {
    const base  = p.severity === 'High' ? 90 : p.severity === 'Medium' ? 60 : 30;
    const bonus = p.distance === 'Very Close' ? 8 : p.distance === 'Close' ? 5 : p.distance === 'Medium' ? 2 : 0;
    out.push({
      score:      base + bonus,
      key:        audioWarnings.potholeKey(p),
      colorClass: p.severity === 'High' ? 'red' : p.severity === 'Medium' ? 'orange' : 'yellow',
      icon:       '⚠️',
      title:      `${p.severity} pothole · ${p.distance}`,
      msg:        p.instructions,
    });
  }

  return out.sort((a, b) => b.score - a.score);
}

function formatClockTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export default function Dashboard() {
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);   // canvas drawn over the video (boxes)
  const captureRef = useRef(null);   // hidden canvas used to grab frames
  const fileInputRef = useRef(null);

  const loopRef     = useRef(false); // analysis loop alive?
  const inFlightRef = useRef(false); // a request is currently pending
  const mutedRef    = useRef(false); // read inside the loop without stale state
  const lastSpokenRef = useRef('');
  const lastHistoryRef = useRef({ key: '', at: 0 });

  const [videoUrl, setVideoUrl]   = useState(null);
  const [playing, setPlaying]     = useState(false);
  const [ended, setEnded]         = useState(false);
  const [latest, setLatest]       = useState(null);
  const [alertHistory, setAlertHistory] = useState([]);
  const [error, setError]         = useState(null);
  const [muted, setMuted]         = useState(false);
  const [latencyMs, setLatencyMs] = useState(0);
  const [dragOver, setDragOver]   = useState(false);

  useEffect(() => {
    mutedRef.current = muted;
    if (muted) { audioWarnings.stop(); window.speechSynthesis?.cancel(); }
  }, [muted]);

  // Stop everything when leaving the page.
  useEffect(() => () => {
    loopRef.current = false;
    audioWarnings.stop();
    window.speechSynthesis?.cancel();
  }, []);

  // ── Frame capture + analysis ──────────────────────────────────────────────
  const drawOverlay = (data) => {
    const v = videoRef.current, c = overlayRef.current;
    if (!v || !c || !v.videoWidth) return;
    const dispW = v.clientWidth, dispH = v.clientHeight;
    if (c.width !== dispW)  c.width = dispW;
    if (c.height !== dispH) c.height = dispH;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);

    const sx = dispW / v.videoWidth, sy = dispH / v.videoHeight;

    // Lane edge lines (the road the vehicle is following) — drawn behind the boxes.
    // Detected lanes = solid green; fixed-ROI fallback (no lines) = dashed amber.
    const lane = data.lane;
    if (lane && lane.confident) {
      ctx.strokeStyle = lane.assumed ? 'rgba(0, 200, 220, 0.85)' : 'rgba(0, 220, 0, 0.9)';
      ctx.lineWidth = lane.assumed ? 2 : 3;
      ctx.setLineDash(lane.assumed ? [10, 8] : []);
      for (const key of ['left', 'right']) {
        const seg = lane[key];
        if (!seg) continue;
        const [x1, y1, x2, y2] = seg;
        ctx.beginPath(); ctx.moveTo(x1 * sx, y1 * sy); ctx.lineTo(x2 * sx, y2 * sy); ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    const box = (b, color, label) => {
      const x = b[0] * sx, y = b[1] * sy, w = (b[2] - b[0]) * sx, h = (b[3] - b[1]) * sy;
      ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
      ctx.font = '600 13px "JetBrains Mono", monospace';
      const pad = 6, tw = ctx.measureText(label).width + pad * 2;
      ctx.fillStyle = color; ctx.fillRect(x, Math.max(0, y - 19), tw, 18);
      ctx.fillStyle = '#06080d'; ctx.fillText(label, x + pad, Math.max(12, y - 6));
    };

    for (const p of data.potholes ?? []) {
      const col = p.severity === 'High' ? '#ef4444' : p.severity === 'Medium' ? '#f97316' : '#eab308';
      box(p.box, col, `${p.severity} ${Math.round(p.confidence * 100)}%`);
    }
    for (const t of data.traffic_lights ?? []) {
      const col = t.color === 'Red' ? '#ef4444' : t.color === 'Yellow' ? '#eab308' : t.color === 'Green' ? '#22c55e' : '#9ca3af';
      box(t.box, col, `${t.color}${t.lane_relevant ? '' : ' adj'}`);
    }
  };

  const speak = (data) => {
    if (mutedRef.current) return;
    const top = buildAlerts(data)[0];
    if (!top || top.score < SPEAK_MIN_SCORE) {
      lastSpokenRef.current = '';   // allow re-announcing when a hazard returns
      return;
    }
    if (top.key === lastSpokenRef.current) return;  // already announced this one
    lastSpokenRef.current = top.key;

    // Try local audio first; fall back to browser speechSynthesis if file not ready
    const played = audioWarnings.play(top.key);
    if (!played && 'speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(top.msg);
      u.rate = 1.05;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
  };

  const recordAlertHistory = (data) => {
    const top = buildAlerts(data)[0];
    if (!top) return;

    const now = Date.now();
    const key = `${top.title}|${top.msg}`;
    const last = lastHistoryRef.current;
    if (last.key === key && now - last.at < HISTORY_REPEAT_GAP_MS) return;

    lastHistoryRef.current = { key, at: now };
    setAlertHistory(prev => [{
      id: now,
      time: formatClockTime(new Date(now)),
      icon: top.icon,
      title: top.title,
      msg: top.msg,
      colorClass: top.colorClass,
    }, ...prev].slice(0, 6));
  };

  const analyzeOnce = async () => {
    const v = videoRef.current, cap = captureRef.current;
    if (!v || !cap || v.readyState < 2 || !v.videoWidth) return;
    cap.width = v.videoWidth; cap.height = v.videoHeight;
    cap.getContext('2d').drawImage(v, 0, 0, cap.width, cap.height);
    const blob = await new Promise(res => cap.toBlob(res, 'image/jpeg', JPEG_QUALITY));
    if (!blob) return;

    const t0 = performance.now();
    const data = await detectImage(blob, 'frame.jpg', 'dashcam');  // use dashcam model (falls back if untrained)
    setLatencyMs(Math.round(performance.now() - t0));
    setLatest(data);
    recordAlertHistory(data);
    drawOverlay(data);
    speak(data);
  };

  const loop = async () => {
    while (loopRef.current) {
      const v = videoRef.current;
      if (v && !v.paused && !v.ended && !inFlightRef.current) {
        inFlightRef.current = true;
        try { await analyzeOnce(); setError(null); }
        catch (e) { setError(e.message || String(e)); }
        finally { inFlightRef.current = false; }
      }
      await new Promise(r => setTimeout(r, ANALYZE_GAP_MS));
    }
  };

  const ensureLoop = () => { if (!loopRef.current) { loopRef.current = true; loop(); } };

  // ── Video lifecycle ───────────────────────────────────────────────────────
  const handlePlay  = () => { setPlaying(true); setEnded(false); ensureLoop(); };
  const handlePause = () => setPlaying(false);
  const handleEnded = () => {
    setPlaying(false); setEnded(true);
    loopRef.current = false; lastSpokenRef.current = '';
    audioWarnings.stop(); window.speechSynthesis?.cancel();
  };

  const togglePlay = () => { const v = videoRef.current; if (v) v.paused ? v.play() : v.pause(); };
  const restart    = () => { const v = videoRef.current; if (v) { v.currentTime = 0; v.play(); } };

  // ── File handling ─────────────────────────────────────────────────────────
  const pickFile = (file) => {
    if (!file || !file.type.startsWith('video/')) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setLatest(null); setAlertHistory([]); lastHistoryRef.current = { key: '', at: 0 };
    setError(null); setEnded(false);
  };
  const changeVideo = () => {
    loopRef.current = false; audioWarnings.stop(); window.speechSynthesis?.cancel();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null); setLatest(null); setAlertHistory([]);
    lastHistoryRef.current = { key: '', at: 0 };
    setError(null); setPlaying(false); setEnded(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Derived display state ─────────────────────────────────────────────────
  const alerts  = buildAlerts(latest);
  const primary = error
    ? { colorClass: 'red', icon: '⛔', title: 'Backend offline',
        msg: `Can't reach ${API_BASE} — start it with: python backend/app.py` }
    : alerts[0]
    ?? (latest
        ? { colorClass: 'green',   icon: '✅', title: 'Road clear', msg: 'No hazards detected ahead.' }
        : { colorClass: 'neutral', icon: '🚗', title: 'Standby',    msg: 'Play the dashcam feed to begin live analysis.' });

  const tl = (latest?.traffic_lights ?? []).find(t => t.lane_relevant)
          ?? (latest?.traffic_lights ?? [])[0] ?? null;
  const previousAlerts = alertHistory.slice(1, 6);

  return (
    <div className="dash">
      {/* Status strip */}
      <div className="dash-status">
        <span className={`dash-live ${playing ? 'on' : ''}`}>
          {playing ? '● LIVE' : ended ? '■ ENDED' : '❚❚ PAUSED'}
        </span>
        <div className="dash-metrics">
          <span>{latencyMs ? `${latencyMs} ms / frame` : 'idle'}</span>
          <span>PH {latest?.counts?.potholes ?? 0} · TL {latest?.counts?.traffic_lights ?? 0}</span>
        </div>
        <button className="dash-mute" onClick={() => setMuted(m => !m)}>
          {muted ? '🔇 Voice off' : '🔊 Voice on'}
        </button>
      </div>

      {!videoUrl ? (
        <div
          className={`dash-upload ${dragOver ? 'drag-over' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); pickFile(e.dataTransfer.files[0]); }}
        >
          <input ref={fileInputRef} type="file" accept="video/*" hidden
                 onChange={(e) => pickFile(e.target.files[0])} />
          <span className="dash-upload-icon">🎥</span>
          <strong>Upload dashcam footage</strong>
          <p>MP4 · WebM · MOV — analysed live, frame by frame</p>
        </div>
      ) : (
        <div className="dash-grid">
          {/* Live feed + overlay */}
          <div className="dash-stage">
            <div className="dash-video-wrap">
              <video
                ref={videoRef}
                src={videoUrl}
                className="dash-video"
                autoPlay muted playsInline
                onPlay={handlePlay}
                onPause={handlePause}
                onEnded={handleEnded}
              />
              <canvas ref={overlayRef} className="dash-overlay" />
            </div>
            <div className="dash-controls">
              <button className="dash-btn primary" onClick={togglePlay}>
                {playing ? '❚❚ Pause' : '▶ Play'}
              </button>
              <button className="dash-btn" onClick={restart}>↻ Restart</button>
              <button className="dash-btn ghost" onClick={changeVideo}>✕ Change video</button>
            </div>
          </div>

          {/* Glanceable alert panel */}
          <aside className={`dash-alert ${primary.colorClass}`}>
            <div className="dash-alert-main">
              <span className="dash-alert-icon">{primary.icon}</span>
              <span className="dash-alert-title">{primary.title}</span>
              <span className="dash-alert-msg">{primary.msg}</span>
            </div>

            {/* Traffic signal indicator */}
            <div className="dash-signal" aria-hidden>
              <span className={`dash-dot red ${tl?.color === 'Red' ? 'lit' : ''}`} />
              <span className={`dash-dot yellow ${tl?.color === 'Yellow' ? 'lit' : ''}`} />
              <span className={`dash-dot green ${tl?.color === 'Green' ? 'lit' : ''}`} />
            </div>

            {/* Secondary hazards */}
            <div className="dash-secondary">
              {alerts.slice(1, 4).map((a, i) => (
                <div key={i} className={`dash-chip ${a.colorClass}`}>
                  <span>{a.icon}</span><span>{a.title}</span>
                </div>
              ))}
              {alerts.length <= 1 && !error && (
                <div className="dash-chip muted">No other hazards</div>
              )}
            </div>

            <div className="dash-history">
              <div className="dash-history-title">Previous alerts</div>
              {previousAlerts.length ? (
                previousAlerts.map(item => (
                  <div key={item.id} className={`dash-history-row ${item.colorClass}`}>
                    <span className="dash-history-time">{item.time}</span>
                    <span className="dash-history-icon">{item.icon}</span>
                    <span className="dash-history-text">{item.title}</span>
                  </div>
                ))
              ) : (
                <div className="dash-history-empty">No previous alerts</div>
              )}
            </div>
          </aside>
        </div>
      )}

      {/* Hidden frame-grab surface */}
      <canvas ref={captureRef} style={{ display: 'none' }} />
    </div>
  );
}
