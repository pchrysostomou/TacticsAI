// TypeScript types for TacticsAI analysis data

export interface VideoInfo {
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  duration_s: number;
  filename: string;
}

export interface TeamSpeed {
  total_dist_m: number;
  team_max_spd_kmh: number;
  top_sprinter_id: string;
}

export interface PressingEvent {
  timestamp: number;
  team_id: number;
  avg_dist_m: number;
}

export interface FramePayload {
  type: 'frame';
  frame_idx: number;
  timestamp: number;
  progress: number;
  total_frames: number;
  player_count: number;
  ball_detected: boolean;
  annotated: string | null;       // base64 JPEG
  pitch_view: string | null;
  heatmap_view: string | null;
  team_counts: Record<string, number>;
  formations: Record<string, string>;
  pressing_state: Record<string, boolean>;
  pressing_events: PressingEvent[];
  space_control: Record<string, number>;
  press_intensity: Record<string, number>;
  teams_spatial: Record<string, { width_m: number; depth_m: number; cohesion: number }>;
  ball_zone: string;
  team_speed: Record<string, TeamSpeed>;
}

export interface CalibrationPayload {
  type: 'calibration';
  src_points: number[][];
}

export interface DonePayload {
  type: 'done';
}

export interface ErrorPayload {
  type: 'error';
  message: string;
}

export type WSPayload = FramePayload | CalibrationPayload | DonePayload | ErrorPayload;

export interface AnalysisSettings {
  skipFrames: number;
  maxFrames: number;
  calibrationFrames: number;
  modelPath: string;
  device: string;
}
