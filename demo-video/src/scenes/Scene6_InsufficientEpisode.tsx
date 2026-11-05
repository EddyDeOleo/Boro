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
  Video,
} from 'remotion';
import { Header } from '../components/Header';
import { SubtitleOverlay } from '../components/SubtitleOverlay';
import { MetricBadge } from '../components/MetricBadge';
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';

export const Scene6_InsufficientEpisode: React.FC = () => {
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
      <Audio src={staticFile('audio/scene6.wav')} />
      <Header sceneNumber="06" sceneTitle="Refusal to Guess: Insufficient Evidence" />

      {/* Main Grid: Left Video & Relative Distance Plot / Right Explainable Verdict */}
      <div
        style={{
          position: 'absolute',
          top: 125,
          left: 60,
          right: 60,
          bottom: 145,
          display: 'grid',
          gridTemplateColumns: '1.15fr 0.85fr',
          gap: '36px',
          alignItems: 'center',
        }}
      >
        {/* Left Column: Stack of Slip Video + Real Analysis Plot */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
            height: '100%',
          }}
        >
          {/* Top: Critical Video Clip */}
          <div
            style={{
              position: 'relative',
              flex: 1.15,
              backgroundColor: '#000000',
              borderRadius: '16px',
              overflow: 'hidden',
              border: `2px solid ${THEME.colors.statusInsufficient}77`,
              boxShadow: '0 10px 30px rgba(0, 0, 0, 0.6)',
            }}
          >
            <Video
              src={staticFile('videos/episode_000025_critical.mp4')}
              playbackRate={0.7}
              loop
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
              }}
            />

            <div
              style={{
                position: 'absolute',
                top: 14,
                left: 14,
                display: 'flex',
                gap: '8px',
              }}
            >
              <MetricBadge type="INSUFFICIENT_EVIDENCE" label="PARTIAL SEGMENT (1.67s)" size="sm" />
              <span
                style={{
                  fontFamily: THEME.fonts.mono,
                  fontSize: '11px',
                  color: THEME.colors.statusInsufficient,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  padding: '3px 8px',
                  borderRadius: '6px',
                  fontWeight: 600,
                }}
              >
                TIMESTAMPS: 5.80s – 7.47s (51 FRAMES)
              </span>
            </div>
          </div>

          {/* Bottom: Real Centroid Relative Distance Plot */}
          <div
            style={{
              position: 'relative',
              flex: 0.85,
              backgroundColor: THEME.colors.bgCard,
              borderRadius: '16px',
              overflow: 'hidden',
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '6px 14px',
              boxSizing: 'border-box',
            }}
          >
            <Img
              src={staticFile('images/relative_distance.png')}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: 8,
                right: 12,
                fontSize: '11px',
                fontFamily: THEME.fonts.mono,
                color: THEME.colors.brandCyan,
                backgroundColor: 'rgba(11, 15, 25, 0.85)',
                padding: '3px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              CENTROID SEPARATION: +12.27 px INCREASE
            </div>
          </div>
        </div>

        {/* Right Column: Criteria Breakdown & Explicit Honesty */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            opacity: entrance,
            transform: `translateY(${interpolate(entrance, [0, 1], [30, 0])}px)`,
          }}
        >
          {/* Verdict Overview Card */}
          <div
            style={{
              backgroundColor: THEME.colors.bgCard,
              border: `1px solid ${THEME.colors.statusInsufficient}55`,
              borderRadius: '16px',
              padding: '20px',
              background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(17, 24, 39, 0.9) 100%)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
                  VERDICT ENGINE OUTCOME
                </div>
                <div style={{ fontSize: '30px', fontWeight: 900, color: THEME.colors.statusInsufficient }}>
                  INSUFFICIENT EVIDENCE
                </div>
              </div>
              <MetricBadge type="INSUFFICIENT_EVIDENCE" label="PARTIAL SEGMENT" size="md" />
            </div>

            <p style={{ fontSize: '13px', color: THEME.colors.textSecondary, margin: '10px 0 0 0', lineHeight: 1.45 }}>
              The available evidence covers only a 1.67s segment. Mandatory target placement cannot be confirmed from this partial observation.
            </p>
          </div>

          {/* Criteria Breakdown */}
          <div
            style={{
              backgroundColor: '#0A0E17',
              border: `1px solid ${THEME.colors.bgCardBorder}`,
              borderRadius: '16px',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
              CRITERIA STATUS FOR OBSERVED WINDOW
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: 'rgba(255, 255, 255, 0.03)',
                padding: '10px 14px',
                borderRadius: '8px',
              }}
            >
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#FFFFFF' }}>Coordinated Transport</div>
                <div style={{ fontSize: '11px', color: THEME.colors.textMuted, fontFamily: THEME.fonts.mono }}>Motion similarity: 0.973</div>
              </div>
              <MetricBadge type="SATISFIED" label="SATISFIED" size="sm" />
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: 'rgba(255, 255, 255, 0.03)',
                padding: '10px 14px',
                borderRadius: '8px',
              }}
            >
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#FFFFFF' }}>Relative Separation</div>
                <div style={{ fontSize: '11px', color: THEME.colors.textMuted, fontFamily: THEME.fonts.mono }}>Centroid distance increased 53px → 65px</div>
              </div>
              <MetricBadge type="SATISFIED" label="SATISFIED" size="sm" />
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: 'rgba(245, 158, 11, 0.08)',
                border: '1px solid rgba(245, 158, 11, 0.25)',
                padding: '10px 14px',
                borderRadius: '8px',
              }}
            >
              <div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: THEME.colors.statusInsufficient }}>
                  Stable Target Placement (Mandatory)
                </div>
                <div style={{ fontSize: '11px', color: THEME.colors.textMuted, fontFamily: THEME.fonts.mono }}>
                  Missing detection inside clip window
                </div>
              </div>
              <MetricBadge type="INSUFFICIENT_EVIDENCE" label="INSUFFICIENT" size="sm" />
            </div>
          </div>

          {/* Honest Edge-Case Limitation Callout */}
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.06)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              padding: '12px 16px',
              borderRadius: '10px',
              fontSize: '12px',
              color: THEME.colors.textSecondary,
              lineHeight: 1.45,
            }}
          >
            <strong style={{ color: '#F87171' }}>Known Nuance & Boundary: </strong>
            Because the cube tumbled into the box after slipping, naive end-state checks falsely flag full episode 25 as Verified.
            Boro&rsquo;s refusal to guess on incomplete intervals prevents ungrounded false positives.
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene6} />
    </AbsoluteFill>
  );
};
