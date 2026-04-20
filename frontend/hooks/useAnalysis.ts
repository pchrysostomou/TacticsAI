'use client';

import { useCallback, useRef, useState } from 'react';
import { AnalysisSettings, FramePayload, WSPayload } from '@/types/analysis';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const WS_API = API.replace('http', 'ws');

export type Status = 'idle' | 'uploading' | 'calibrating' | 'analyzing' | 'done' | 'error';

/** Lightweight stats that drive React re-renders (no images). */
export interface FrameStats {
  frame_idx:      number;
  timestamp:      number;
  progress:       number;
  total_frames:   number;
  player_count:   number;
  ball_detected:  boolean;
  team_counts:    Record<string, number>;
  formations:     Record<string, string>;
  pressing_state: Record<string, boolean>;
  pressing_events: Array<{ timestamp: number; team_id: number; avg_dist_m: number }>;
  space_control:  Record<string, number>;
  teams_spatial:  Record<string, { width_m: number; depth_m: number; cohesion: number }>;
  ball_zone:      string;
  team_speed:     Record<string, { total_dist_m: number; team_max_spd_kmh: number; top_sprinter_id: string }>;
}

function extractStats(p: FramePayload): FrameStats {
  return {
    frame_idx:      p.frame_idx,
    timestamp:      p.timestamp,
    progress:       p.progress,
    total_frames:   p.total_frames,
    player_count:   p.player_count,
    ball_detected:  p.ball_detected,
    team_counts:    p.team_counts,
    formations:     p.formations,
    pressing_state: p.pressing_state,
    pressing_events: p.pressing_events ?? [],
    space_control:  p.space_control ?? {},
    teams_spatial:  p.teams_spatial ?? {},
    ball_zone:      p.ball_zone ?? '—',
    team_speed:     p.team_speed ?? {},
  };
}

export interface AnalysisRefs {
  annotatedRef: React.RefObject<HTMLImageElement | null>;
  pitchRef:     React.RefObject<HTMLImageElement | null>;
  heatmapRef:   React.RefObject<HTMLImageElement | null>;
}

export function useAnalysis() {
  const [status,  setStatus ] = useState<Status>('idle');
  const [error,   setError  ] = useState<string | null>(null);
  const [stats,   setStats  ] = useState<FrameStats | null>(null);
  const [srcPts,  setSrcPts ] = useState<number[][] | null>(null);

  // Image elements updated directly — zero React re-renders for heavy data
  const annotatedRef = useRef<HTMLImageElement | null>(null);
  const pitchRef     = useRef<HTMLImageElement | null>(null);
  const heatmapRef   = useRef<HTMLImageElement | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // Throttle React state updates to max 10 fps (stats only)
  const lastStatsUpdate = useRef(0);

  const upload = useCallback(async (file: File): Promise<string | null> => {
    setStatus('uploading');
    setError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      return data.session_id as string;
    } catch (e: any) {
      setError(e.message);
      setStatus('error');
      return null;
    }
  }, []);

  const start = useCallback(async (file: File, settings: AnalysisSettings) => {
    wsRef.current?.close();

    const sessionId = await upload(file);
    if (!sessionId) return;

    const qs = new URLSearchParams({
      skip_frames:        String(settings.skipFrames),
      max_frames:         String(settings.maxFrames),
      calibration_frames: String(settings.calibrationFrames),
      model_path:         settings.modelPath,
      device:             settings.device,
    });

    const ws = new WebSocket(`${WS_API}/ws/${sessionId}?${qs}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const payload: WSPayload = JSON.parse(e.data);

      switch (payload.type) {
        case 'calibration':
          setStatus('calibrating');
          setSrcPts(payload.src_points);
          break;

        case 'frame': {
          // ── Direct DOM updates for images (no React overhead) ──
          if (payload.annotated && annotatedRef.current) {
            annotatedRef.current.src = `data:image/jpeg;base64,${payload.annotated}`;
          }
          if (payload.pitch_view && pitchRef.current) {
            pitchRef.current.src = `data:image/jpeg;base64,${payload.pitch_view}`;
          }
          if (payload.heatmap_view && heatmapRef.current) {
            heatmapRef.current.src = `data:image/jpeg;base64,${payload.heatmap_view}`;
          }

          // ── Throttled React state update for lightweight stats ──
          const now = Date.now();
          if (now - lastStatsUpdate.current > 100) {   // max 10 fps for React
            lastStatsUpdate.current = now;
            setStatus('analyzing');
            setStats(extractStats(payload));
          }
          break;
        }

        case 'done':
          setStatus('done');
          // Final stats update without throttle
          break;

        case 'error':
          setError(payload.message);
          setStatus('error');
          break;
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
      setStatus('error');
    };

    ws.onclose = () =>
      setStatus(s => (s === 'analyzing' || s === 'calibrating') ? 'done' : s);

    setStatus('calibrating');
  }, [upload]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setStatus('idle');
  }, []);

  return {
    status,
    error,
    stats,
    srcPts,
    start,
    stop,
    refs: { annotatedRef, pitchRef, heatmapRef } as AnalysisRefs,
  };
}
