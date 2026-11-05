export type VerdictType = 'VERIFIED' | 'FAILED' | 'INSUFFICIENT_EVIDENCE';

export type CoverageScopeType = 'COMPLETE_TASK' | 'PARTIAL_SEGMENT' | 'NONE';

export interface BoroOverview {
  verdict: VerdictType;
  summary: string;
  coverage_scope: CoverageScopeType;
  covers_entire_context: boolean;
  start_time_seconds: number;
  end_time_seconds: number;
  duration_seconds: number;
  fps: number;
  context_frame_count: number;
  observed_frame_count: number;
}

export interface BoroCriterion {
  criterion_id: string;
  requirement: 'REQUIRED' | 'FORBIDDEN' | 'OPTIONAL';
  is_mandatory: boolean;
  status: 'SATISFIED' | 'VIOLATED' | 'INSUFFICIENT_EVIDENCE';
  explanation: string;
  event_ids?: string[];
  evidence_ids?: string[];
}

export interface BoroTimelineEvent {
  event_id: string;
  event_type: string;
  status: 'DETECTED' | 'NOT_DETECTED';
  frame_start: number;
  frame_end: number;
  timestamp_start: number;
  timestamp_end: number;
  confidence: number;
  attributes?: Record<string, unknown>;
}

export interface SceneConfig {
  id: string;
  title: string;
  subtitle: string;
  durationInFrames: number;
}
