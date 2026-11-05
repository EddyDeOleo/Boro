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

export const Scene7_FrontendPlatform: React.FC = () => {
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
      <Audio src={staticFile('audio/scene7.wav')} />
      <Header sceneNumber="07" sceneTitle="Interactive Operator Experience" />

      {/* Main Grid: Left Real Frontend Video Clip / Right Workflow Transparency */}
      <div
        style={{
          position: 'absolute',
          top: 125,
          left: 60,
          right: 60,
          bottom: 145,
          display: 'grid',
          gridTemplateColumns: '1.2fr 0.8fr',
          gap: '36px',
          alignItems: 'center',
        }}
      >
        {/* Left: Video Showing Configuration & Polygon Marking */}
        <div
          style={{
            position: 'relative',
            height: '100%',
            backgroundColor: '#000000',
            borderRadius: '20px',
            overflow: 'hidden',
            border: `1px solid ${THEME.colors.bgCardBorder}`,
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.7)',
          }}
        >
          {/* Loop through the configuration segment in frontend demo (approx frames 1500 to 2200) */}
          <Video
            src={staticFile('videos/boro_frontend_demo.mp4')}
            startFrom={1500}
            endAt={2200}
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
            <MetricBadge type="INFO" label="BORO WORKSPACE" size="sm" />
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
              VISUAL TARGET & REGION CONFIGURATION
            </span>
          </div>

          <div
            style={{
              position: 'absolute',
              bottom: 20,
              left: 20,
              right: 20,
              background: 'rgba(11, 15, 25, 0.9)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '12px 18px',
              borderRadius: '12px',
              fontSize: '12px',
              fontFamily: THEME.fonts.mono,
              color: THEME.colors.textSecondary,
              display: 'flex',
              justifyContent: 'space-between',
            }}
          >
            <span>ENTITY 1: cube • ENTITY 2: gripper</span>
            <span style={{ color: THEME.colors.statusVerified }}>REGION 1: target_zone [Configured]</span>
          </div>
        </div>

        {/* Right: Auditable Limitations & Human Verification */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '18px',
            opacity: entrance,
            transform: `translateX(${interpolate(entrance, [0, 1], [30, 0])}px)`,
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
              Human-in-the-Loop Transparency
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
              Auditable Evidence & Explicit Boundaries
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
            Boro does not hide uncertainty behind opaque probability scores.
            Every generated report itemizes explicit physical and visual limitations directly alongside
            the evidence.
          </p>

          {/* Transparent Limitations List */}
          <div
            style={{
              backgroundColor: '#0A0E17',
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              borderRadius: '16px',
              padding: '18px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
              LOGGED REPORT LIMITATIONS
            </div>

            {[
              'Displacement measured in 2D image coordinates, not physical 3D space.',
              'Visual mask proximity does not establish tactile force or grip stability.',
              'Perspective, camera movement, and occlusions are accounted for in coverage declarations.',
              'Absence of an event detection does not prove it did not occur.',
            ].map((lim, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  fontSize: '13px',
                  color: THEME.colors.textSecondary,
                  lineHeight: 1.4,
                }}
              >
                <span style={{ color: THEME.colors.statusInsufficient, fontWeight: 700 }}>•</span>
                <span>{lim}</span>
              </div>
            ))}
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.04)',
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              borderRadius: '12px',
              padding: '14px 18px',
              fontSize: '13px',
              color: THEME.colors.textPrimary,
              fontWeight: 600,
            }}
          >
            Execution Coverage Scope: <span style={{ color: THEME.colors.brandCyan }}>Verified Against Whole Video Context</span>
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene7} />
    </AbsoluteFill>
  );
};
