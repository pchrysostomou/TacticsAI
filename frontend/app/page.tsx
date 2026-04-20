'use client';

import { useCallback, useRef, useState } from 'react';
import { useAnalysis, Status, FrameStats } from '@/hooks/useAnalysis';
import { AnalysisSettings } from '@/types/analysis';

const STATUS_LABEL: Record<Status, string> = {
  idle:        '● Idle',
  uploading:   '↑ Uploading…',
  calibrating: '◈ Calibrating…',
  analyzing:   '▶ Analyzing',
  done:        '✓ Done',
  error:       '✗ Error',
};

const ROADMAP = [
  { label: 'Detection (YOLOv8)',   done: true  },
  { label: 'Tracking (ByteTrack)', done: true  },
  { label: 'Team classification',  done: true  },
  { label: "Bird's-eye view",      done: true  },
  { label: 'Formation detection',  done: true  },
  { label: 'Heatmaps',             done: true  },
  { label: 'Tactical metrics',     done: true  },
  { label: 'Pass network',         done: true  },
  { label: 'Pressing triggers',    done: true  },
  { label: 'Speed tracking',       done: true  },
  { label: 'Player re-ID',         done: true  },
  { label: 'Fine-tuned YOLOv8',    done: false },
];

const TEAM_COLORS = ['#60a5fa', '#4ade80'];

export default function Dashboard() {
  const { status, error, stats, start, stop, refs } = useAnalysis();

  const [settings, setSettings] = useState<AnalysisSettings>({
    skipFrames: 2,
    maxFrames: 300,
    calibrationFrames: 60,
    modelPath: 'yolov8x.pt',
    device: 'cuda',
  });

  const [file,     setFile    ] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [tab,      setTab     ] = useState<'heatmap' | 'metrics'>('metrics');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onFile = (f: File) => setFile(f);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, []);

  const isRunning = status === 'uploading' || status === 'calibrating' || status === 'analyzing';
  const progress  = stats?.progress ?? 0;

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="header">
        <div className="header__logo">⚽</div>
        <span className="header__title gradient-text">TacticsAI</span>
        <div className="header__spacer" />
        {error && (
          <span style={{ fontSize: '0.7rem', color: '#ef4444', maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={error}>
            {error}
          </span>
        )}
        <span className={`header__status header__status--${status}`}>
          {STATUS_LABEL[status]}
        </span>
      </header>

      {/* ── Sidebar ─────────────────────────────────────────────── */}
      <aside className="sidebar">

        {/* Upload */}
        <div>
          <p className="label" style={{ marginBottom: 8 }}>Video</p>
          <div
            className={`upload ${dragging ? 'upload--dragging' : ''}`}
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload__icon">🎬</div>
            <div className="upload__text">
              {file ? file.name : 'Drop video here or click to upload'}
            </div>
            {file && (
              <div className="upload__file">
                {(file.size / 1_048_576).toFixed(1)} MB
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && onFile(e.target.files[0])}
            />
          </div>
        </div>

        <div className="divider" />

        {/* Settings */}
        <div>
          <p className="label" style={{ marginBottom: 10 }}>Processing</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

            <div className="setting">
              <div className="setting__row">
                <span className="setting__label">Frame skip</span>
                <span className="setting__val">{settings.skipFrames}</span>
              </div>
              <input type="range" min={0} max={5} value={settings.skipFrames}
                onChange={e => setSettings(s => ({ ...s, skipFrames: +e.target.value }))} />
            </div>

            <div className="setting">
              <span className="setting__label">Max frames (0 = full)</span>
              <input type="number" min={0} max={10000} value={settings.maxFrames}
                onChange={e => setSettings(s => ({ ...s, maxFrames: +e.target.value }))} />
            </div>

            <div className="setting">
              <div className="setting__row">
                <span className="setting__label">Calibration frames</span>
                <span className="setting__val">{settings.calibrationFrames}</span>
              </div>
              <input type="range" min={20} max={120} value={settings.calibrationFrames}
                onChange={e => setSettings(s => ({ ...s, calibrationFrames: +e.target.value }))} />
            </div>

            <div className="setting">
              <span className="setting__label">Model weights</span>
              <select value={settings.modelPath}
                onChange={e => setSettings(s => ({ ...s, modelPath: e.target.value }))}>
                <option value="yolov8x.pt">yolov8x (max accuracy)</option>
                <option value="yolov8l.pt">yolov8l (large)</option>
                <option value="yolov8m.pt">yolov8m (medium)</option>
                <option value="best.pt">best.pt (fine-tuned)</option>
              </select>
            </div>

            <div className="setting">
              <span className="setting__label">Device</span>
              <select value={settings.device}
                onChange={e => setSettings(s => ({ ...s, device: e.target.value }))}>
                <option value="cuda">CUDA (GPU)</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
          </div>
        </div>

        <div className="divider" />

        {/* Buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button
            className="btn btn--primary"
            disabled={!file || isRunning}
            onClick={() => file && start(file, settings)}
          >
            {isRunning ? `▶ ${STATUS_LABEL[status]}` : '▶ Run Analysis'}
          </button>
          {isRunning && (
            <button className="btn btn--danger" onClick={stop}>■ Stop</button>
          )}
        </div>

        <div className="divider" />

        {/* Roadmap */}
        <div>
          <p className="label" style={{ marginBottom: 8 }}>Roadmap</p>
          <div className="roadmap">
            {ROADMAP.map(({ label, done }) => (
              <div key={label} className="roadmap__item">
                <span className={done ? 'roadmap__check' : 'roadmap__circle'}>
                  {done ? '✓' : '○'}
                </span>
                <span className={done ? 'roadmap__done' : 'roadmap__pending'}>{label}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────── */}
      <main className="main">

        {/* Annotated video — spans 2 rows */}
        <div className="panel panel--span2">
          <div className="panel__header">
            <Dot color="#4B7BF5" glow />
            <span className="panel__title">Annotated Output</span>
            {stats && (
              <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: 'rgba(232,234,240,0.35)' }}>
                Frame {stats.frame_idx} · {stats.timestamp.toFixed(1)}s
              </span>
            )}
          </div>
          <div className="panel__content">
            {/* img element updated directly via ref — no React re-render on each frame */}
            <img
              ref={refs.annotatedRef}
              alt="Annotated frame"
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: isRunning || stats ? 'block' : 'none' }}
            />
            {!isRunning && !stats && (
              <Placeholder icon="🎥" text="Upload a video and click Run Analysis" />
            )}
            {isRunning && !stats && (
              <Placeholder icon="⏳" text={STATUS_LABEL[status]} />
            )}
          </div>
          {/* Progress bar */}
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
        </div>

        {/* Bird's-eye */}
        <div className="panel">
          <div className="panel__header">
            <Dot color="#22c55e" glow />
            <span className="panel__title">Bird's-eye View</span>
            {stats && (
              <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: 'rgba(232,234,240,0.35)' }}>
                {stats.player_count} players
              </span>
            )}
          </div>
          <div className="panel__content">
            <img
              ref={refs.pitchRef}
              alt="Bird's-eye view"
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: isRunning || stats ? 'block' : 'none' }}
            />
            {!isRunning && !stats && <Placeholder icon="🏟️" text="2D pitch view" />}
          </div>
        </div>

        {/* Heatmap + Metrics */}
        <div className="panel" style={{ overflow: 'hidden' }}>
          {/* Tab bar */}
          <div className="panel__header" style={{ gap: 0, padding: 0 }}>
            {(['heatmap', 'metrics'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                flex: 1, padding: '10px 0', background: 'none', border: 'none',
                borderBottom: tab === t ? '2px solid #4B7BF5' : '2px solid transparent',
                color: tab === t ? '#4B7BF5' : 'rgba(232,234,240,0.4)',
                fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.08em',
                textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'inherit',
                transition: '150ms ease',
              }}>
                {t === 'heatmap' ? 'Cumulative Heatmap' : 'Tactical Metrics'}
              </button>
            ))}
          </div>

          <div className="panel__content" style={{ alignItems: 'flex-start', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
            {/* Heatmap image — always rendered; hidden when tab !== heatmap */}
            <img
              ref={refs.heatmapRef}
              alt="Heatmap"
              style={{
                width: '100%', height: '100%', objectFit: 'contain',
                display: tab === 'heatmap' && (isRunning || stats) ? 'block' : 'none',
              }}
            />
            {tab === 'heatmap' && !isRunning && !stats && <Placeholder icon="🔥" text="Cumulative heatmap" />}

            {/* Metrics — React-rendered, throttled */}
            {tab === 'metrics' && <MetricsPanel stats={stats} />}
          </div>
        </div>
      </main>

      {/* ── Stats bar ───────────────────────────────────────────── */}
      <footer className="statsbar">
        <Stat label="Frame"   value={stats?.frame_idx    ?? '—'} />
        <Stat label="Players" value={stats?.player_count  ?? '—'} />
        <Stat label="Ball"    value={stats ? (stats.ball_detected ? 'Yes' : 'No') : '—'} />
        <Stat label="Time"    value={stats ? `${stats.timestamp.toFixed(1)}s` : '—'} />
        <Stat label="Team A"  value={stats?.team_counts?.['0'] ?? '—'} cls="stat__value--a" />
        <Stat label="Team B"  value={stats?.team_counts?.['1'] ?? '—'} cls="stat__value--b" />
        <Stat label="Form. A" value={stats?.formations?.['0']  ?? '—'} />
        <Stat label="Form. B" value={stats?.formations?.['1']  ?? '—'} />
        <Stat label="Space A" value={stats ? `${stats.space_control?.['0'] ?? 0}%` : '—'} cls="stat__value--a" />
        <Stat label="Space B" value={stats ? `${stats.space_control?.['1'] ?? 0}%` : '—'} cls="stat__value--b" />
        {status === 'done' && stats && (
          <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: '#22c55e', fontWeight: 700 }}>
            ✓ Complete — {stats.frame_idx} frames processed
          </span>
        )}
      </footer>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────

function Dot({ color, glow }: { color: string; glow?: boolean }) {
  return (
    <div style={{
      width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0,
      ...(glow ? { boxShadow: `0 0 6px ${color}` } : {}),
    }} />
  );
}

function Placeholder({ icon, text }: { icon: string; text: string }) {
  return (
    <div className="placeholder">
      <div className="placeholder__icon">{icon}</div>
      <div>{text}</div>
    </div>
  );
}

function Stat({ label, value, cls }: { label: string; value: string | number; cls?: string }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className={`stat__value ${cls ?? ''}`}>{String(value)}</span>
    </div>
  );
}



function MetricTeamRow({ tid, stats }: { tid: number; stats: FrameStats | null }) {
  const spd     = stats?.team_speed?.[String(tid)];
  const spatial = stats?.teams_spatial?.[String(tid)];
  const press   = stats?.pressing_state?.[String(tid)];
  const space   = stats?.space_control?.[String(tid)];
  const form    = stats?.formations?.[String(tid)];
  const count   = stats?.team_counts?.[String(tid)] ?? 0;

  return (
    <div className="metric-team">
      <div className="metric-team__header">
        <div className="metric-team__dot" style={{ background: TEAM_COLORS[tid] }} />
        <span>Team {tid === 0 ? 'A' : 'B'}</span>
        <span style={{ color: 'rgba(232,234,240,0.4)', fontWeight: 400, fontSize: '0.73rem' }}>
          · {count} {count === 1 ? 'player' : 'players'}
        </span>
        {press && <span className="metric-team__press-badge">PRESSING</span>}
      </div>
      <div className="metric-row">
        <MetricItem label="Formation"   value={form ?? '—'} />
        <MetricItem label="Space ctrl"  value={`${space ?? 0}%`} />
        <MetricItem label="Width"       value={`${spatial?.width_m  ?? 0} m`} />
        <MetricItem label="Depth"       value={`${spatial?.depth_m  ?? 0} m`} />
        <MetricItem label="Dist run"    value={`${spd?.total_dist_m      ?? 0} m`} />
        <MetricItem label="Max speed"   value={`${spd?.team_max_spd_kmh  ?? 0} km/h`} />
      </div>
    </div>
  );
}

function MetricsPanel({ stats }: { stats: FrameStats | null }) {
  if (!stats) return <Placeholder icon="📊" text="Tactical metrics" />;

  return (
    <div className="metrics" style={{ width: '100%' }}>
      <MetricTeamRow tid={0} stats={stats} />
      <MetricTeamRow tid={1} stats={stats} />

      <div style={{ fontSize: '0.72rem', color: 'rgba(232,234,240,0.45)', paddingTop: 4 }}>
        Ball zone: <strong style={{ color: '#e8eaf0' }}>{stats.ball_zone}</strong>
      </div>

      {stats.pressing_events.length > 0 && (
        <div>
          <p className="label" style={{ marginBottom: 6 }}>Pressing Events</p>
          <div className="events">
            {stats.pressing_events.map((evt, i) => (
              <div key={i} className="event">
                t={evt.timestamp.toFixed(1)}s — Team {evt.team_id === 0 ? 'A' : 'B'} @ {evt.avg_dist_m} m
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-item">
      <span className="metric-item__label">{label}</span>
      <span className="metric-item__value">{value}</span>
    </div>
  );
}
