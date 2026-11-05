import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import { Header } from '../components/Header';
import { SubtitleOverlay } from '../components/SubtitleOverlay';
import { MetricBadge } from '../components/MetricBadge';
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';

export const Scene4_SAMTracking: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.8 },
  });

  // Cycle through real overlay frames or show dual comparison
  const showDetail = frame > 360;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: THEME.colors.bgDark,
        fontFamily: THEME.fonts.heading,
        color: THEME.colors.textPrimary,
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('audio/scene4.wav')} />
      <Header sceneNumber="04" sceneTitle="Meta SAM 3.1 Visual Tracking" />

      {/* Main Grid: Left Tracking Visuals / Right Perception Engine Breakdown */}
      <div
        style={{
          position: 'absolute',
          top: 125,
          left: 60,
          right: 60,
          bottom: 145,
          display: 'grid',
          gridTemplateColumns: '1.25fr 0.75fr',
          gap: '36px',
          alignItems: 'center',
        }}
      >
        {/* Left: Real Segmentation Evidence Image Frame */}
        <div
          style={{
            position: 'relative',
            height: '100%',
            backgroundColor: '#000000',
            borderRadius: '20px',
            overflow: 'hidden',
            border: `1px solid ${THEME.colors.brandCyan}44`,
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Img
            src={staticFile(
              showDetail
                ? 'images/dual_tracking_comparison.jpg'
                : 'images/episode_000025_tracking_frames.jpg'
            )}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              transform: `scale(${interpolate(frame, [0, 750], [1, 1.04])})`,
            }}
          />

          {/* Top HUD Badges */}
          <div
            style={{
              position: 'absolute',
              top: 20,
              left: 20,
              display: 'flex',
              gap: '10px',
            }}
          >
            <MetricBadge type="INFO" label="META SAM 3.1" size="sm" />
            <span
              style={{
                fontFamily: THEME.fonts.mono,
                fontSize: '12px',
                color: THEME.colors.brandCyan,
                backgroundColor: 'rgba(0,0,0,0.7)',
                padding: '4px 10px',
                borderRadius: '6px',
                fontWeight: 600,
              }}
            >
              DUAL MASK TRACKING • 51 OBSERVED FRAMES
            </span>
          </div>

          {/* Bottom HUD: Dynamic mask coordinates overlay */}
          <div
            style={{
              position: 'absolute',
              bottom: 20,
              left: 20,
              right: 20,
              background: 'rgba(11, 15, 25, 0.88)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              padding: '12px 18px',
              borderRadius: '12px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: THEME.fonts.mono,
              fontSize: '13px',
            }}
          >
            <div>
              <span style={{ color: THEME.colors.textMuted }}>OBJECT ID 1: </span>
              <span style={{ color: '#EF4444', fontWeight: 700 }}>red_cube</span>
              <span style={{ color: THEME.colors.textMuted, marginLeft: '16px' }}>EFFECTOR ID 2: </span>
              <span style={{ color: THEME.colors.brandCyan, fontWeight: 700 }}>gripper</span>
            </div>
            <div style={{ color: THEME.colors.statusVerified }}>
              TRACKING CONFIDENCE: 0.978
            </div>
          </div>
        </div>

        {/* Right: Technical Explanation & Disclaimers */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '20px',
            opacity: entrance,
            transform: `translateY(${interpolate(entrance, [0, 1], [30, 0])}px)`,
          }}
        >
          <div>
            <div
              style={{
                color: THEME.colors.brandCyan,
                fontSize: '13px',
                fontFamily: THEME.fonts.mono,
                fontWeight: 700,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                marginBottom: '8px',
              }}
            >
              Empirical Tracking Layer
            </div>
            <h2
              style={{
                fontSize: '34px',
                fontWeight: 800,
                lineHeight: 1.2,
                letterSpacing: '-0.02em',
                margin: 0,
              }}
            >
              Prompt-Based Video Segmentation
            </h2>
          </div>

          <p
            style={{
              fontSize: '16px',
              color: THEME.colors.textSecondary,
              lineHeight: 1.5,
              margin: 0,
            }}
          >
            Starting from user-provided initial bounding boxes, Meta SAM 3.1 isolates and tracks
            manipulated entities across the video sequence, generating pixel-accurate masks for downstream
            geometric analysis.
          </p>

          {/* Metric cards extracted from evaluation */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '12px',
            }}
          >
            <div
              style={{
                backgroundColor: THEME.colors.bgCard,
                border: `1px solid ${THEME.colors.bgCardBorder}`,
                borderRadius: '12px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '11px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
                MEASUREMENT TYPE
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '4px', color: '#FFFFFF' }}>
                Centroid Displacement
              </div>
              <div style={{ fontSize: '12px', color: THEME.colors.textSecondary, marginTop: '4px' }}>
                2D Image Space
              </div>
            </div>

            <div
              style={{
                backgroundColor: THEME.colors.bgCard,
                border: `1px solid ${THEME.colors.bgCardBorder}`,
                borderRadius: '12px',
                padding: '16px',
              }}
            >
              <div style={{ fontSize: '11px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
                MOTION SIMILARITY
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '4px', color: THEME.colors.brandCyan }}>
                0.973 Mean Coordinated
              </div>
              <div style={{ fontSize: '12px', color: THEME.colors.textSecondary, marginTop: '4px' }}>
                Geometric Heuristic
              </div>
            </div>
          </div>

          {/* Vital Technical Limitation Box */}
          <div
            style={{
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              padding: '14px 18px',
              fontSize: '13px',
              color: THEME.colors.textSecondary,
              lineHeight: 1.45,
            }}
          >
            <strong style={{ color: THEME.colors.statusInsufficient }}>Physical Limitation Disclosure: </strong>
            Mask proximity and visual motion similarity do not prove physical contact, grip force,
            or suction. Measurements represent 2D image coordinates without 3D tactile sensing.
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene4} />
    </AbsoluteFill>
  );
};
