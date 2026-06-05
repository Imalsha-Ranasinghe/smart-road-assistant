// Local audio warning player.
// Audio files must be placed in public/audio/ as <key>.mp3
// Falls back gracefully when a file hasn't loaded yet.

const AUDIO_DIR = '/audio/';

const KEYS = [
  'ph_high_very_close', 'ph_high_close', 'ph_high_medium', 'ph_high_far',
  'ph_medium_very_close', 'ph_medium_close', 'ph_medium_medium', 'ph_medium_far',
  'ph_low_very_close', 'ph_low_close', 'ph_low_medium', 'ph_low_far',
  'ph_fallback',
  'tl_red_lane', 'tl_red_adjacent',
  'tl_yellow_lane', 'tl_yellow_adjacent',
  'tl_green_lane', 'tl_green_adjacent',
  'tl_unknown_lane', 'tl_unknown_adjacent',
  'tl_fallback',
];

// Preload all files; track which ones actually loaded successfully.
const _pool  = {};
const _ready = new Set();

for (const key of KEYS) {
  const a = new Audio();
  a.preload = 'auto';
  a.addEventListener('canplaythrough', () => _ready.add(key), { once: true });
  a.src = `${AUDIO_DIR}${key}.mp3`;
  _pool[key] = a;
}

let _currentKey = null;
let _current    = null;

/**
 * Play the audio clip for the given key.
 * Returns true  — file was ready and playback started.
 * Returns false — file not loaded; caller should use speechSynthesis fallback.
 */
export function play(key) {
  if (!key || !_ready.has(key)) return false;
  if (_currentKey === key && _current && !_current.paused && !_current.ended) return true;
  stop();
  const a = _pool[key];
  a.currentTime = 0;
  a.play().catch(() => {});
  _currentKey = key;
  _current    = a;
  return true;
}

/** Stop any currently playing warning immediately. */
export function stop() {
  if (_current) {
    _current.pause();
    _current.currentTime = 0;
  }
  _currentKey = null;
  _current    = null;
}

/** Derive the audio key from a pothole detection object. */
export function potholeKey(p) {
  const dist = p.distance.toLowerCase().replace(/\s+/g, '_');
  return `ph_${p.severity.toLowerCase()}_${dist}`;
}

/** Derive the audio key from a traffic-light detection object. */
export function trafficKey(t) {
  return `tl_${t.color.toLowerCase()}_${t.lane_relevant ? 'lane' : 'adjacent'}`;
}
