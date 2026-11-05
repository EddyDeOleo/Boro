import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  Video,
} from 'remotion';
import { Header } from '../components/Header';
import { SubtitleOverlay } from '../components/SubtitleOverlay';
import { MetricBadge } from '../components/MetricBadge';
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';

export const Scene3_MuseSpark: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.8 },
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: THEME.colors.bgDark,
        fontFamily: THEME.fonts.heading,
        color: THEME.colors.textPrimary,
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('audio/scene3.wav')} />
      <Header sceneNumber="03" sceneTitle="Natural-Language Task Interpretation" />

      {/* Main Grid: Left Video UI Demo / Right Muse Spark Structured Schema */}
      <div
        style={{
          position: 'absolute',
          top: 125,
          left: 60,
          right: 60,
          bottom: 145,
          display: 'grid',
          gridTemplateColumns: '1.1fr 0.9fr',
          gap: '36px',
          alignItems: 'center',
        }}
      >
        {/* Left: Real Frontend Demo Clip */}
        <div
          style={{
            position: 'relative',
            height: '100%',
            backgroundColor: '#000000',
            borderRadius: '20px',
            overflow: 'hidden',
            border: `1px solid ${THEME.colors.bgCardBorder}`,
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.7)',
          }}
        >
          <Video
            src={staticFile('videos/boro_frontend_demo.mp4')}
            startFrom={600} // Frame 600 (~20s into demo where prompt & criteria are displayed)
            endAt={1350}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />

          <div
            style={{
              position: 'absolute',
              top: 20,
              left: 20,
              display: 'flex',
              gap: '10px',
            }}
          >
            <MetricBadge type="MANDATORY" label="META MUSE SPARK" size="sm" />
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
              TASK SPECIFICATION PIPELINE
            </span>
          </div>
        </div>

        {/* Right: Structured Output Presentation */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '20px',
            opacity: entrance,
            transform: `translateX(${interpolate(entrance, [0, 1], [30, 0])}px)`,
          }}
        >
          {/* User Prompt Box */}
          <div
            style={{
              backgroundColor: THEME.colors.bgCard,
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              borderRadius: '16px',
              padding: '20px',
            }}
          >
            <div
              style={{
                fontSize: '12px',
                fontWeight: 700,
                color: THEME.colors.brandCyanLight,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginBottom: '8px',
                fontFamily: THEME.fonts.mono,
              }}
            >
              Natural-Language Instruction Input
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: 600,
                color: '#FFFFFF',
                lineHeight: 1.4,
              }}
            >
              &ldquo;Pick up the cube, transport it to the target zone, separate from it, and leave it stable inside the target zone.&rdquo;
            </div>
          </div>

          {/* Structured Output Roles & Criteria */}
          <div
            style={{
              backgroundColor: '#0A0E17',
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              borderRadius: '16px',
              padding: '20px',
            }}
          >
            <div
              style={{
                fontSize: '12px',
                fontWeight: 700,
                color: THEME.colors.brandCyan,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginBottom: '14px',
                fontFamily: THEME.fonts.mono,
              }}
            >
              Muse Spark Semantic Roles & Criteria Schema
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[
                { id: 'object_transport_with_effector', label: 'Coordinated Effector Transport', mandatory: true },
                { id: 'object_reaches_target', label: 'Object Enters Target Region', mandatory: true },
                { id: 'object_separates_from_effector', label: 'Effector Disengagement / Separation', mandatory: true },
                { id: 'object_stable_in_target', label: 'Stable Placement in Target Zone', mandatory: true },
              ].map((c) => (
                <div
                  key={c.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    backgroundColor: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                  }}
                >
                  <span style={{ fontSize: '14px', fontWeight: 600, color: '#FFFFFF' }}>
                    {c.label}
                  </span>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
                      {c.id}
                    </span>
                    <MetricBadge type="MANDATORY" label="MANDATORY" size="sm" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Technical Boundary Disclaimer */}
          <div
            style={{
              background: 'rgba(6, 104, 225, 0.08)',
              border: '1px solid rgba(6, 104, 225, 0.25)',
              padding: '12px 18px',
              borderRadius: '10px',
              fontSize: '13px',
              color: THEME.colors.textSecondary,
              lineHeight: 1.45,
            }}
          >
            <strong style={{ color: THEME.colors.brandCyanLight }}>Architectural Boundary: </strong>
            Muse Spark is dedicated solely to linguistic interpretation and formal criteria generation.
            It does not evaluate videos or declare verdicts.
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene3} />
    </AbsoluteFill>
  );
};
