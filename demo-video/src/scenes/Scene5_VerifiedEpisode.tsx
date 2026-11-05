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

export const Scene5_VerifiedEpisode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.8 },
  });

  // Calculate simulated timeline progress across the 10.0s episode 000000
  const videoProgressSec = (frame / 750) * 10.0;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: THEME.colors.bgDark,
        fontFamily: THEME.fonts.heading,
        color: THEME.colors.textPrimary,
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('audio/scene5.wav')} />
      <Header sceneNumber="05" sceneTitle="Verified Ground-Truth Execution" />

      {/* Main Grid: Left Real Video / Right Verified Report & Timeline */}
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
        {/* Left: Video Player with Real Episode 000000 Footage */}
        <div
          style={{
            position: 'relative',
            height: '100%',
            backgroundColor: '#000000',
            borderRadius: '20px',
            overflow: 'hidden',
            border: `2px solid ${THEME.colors.statusVerified}66`,
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.7), 0 0 30px rgba(16, 185, 129, 0.2)',
          }}
        >
          <Video
            src={staticFile('videos/episode_000000.mp4')}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />

          {/* Top HUD */}
          <div
            style={{
              position: 'absolute',
              top: 20,
              left: 20,
              display: 'flex',
              gap: '10px',
            }}
          >
            <MetricBadge type="VERIFIED" label="VERIFIED EXECUTION" size="sm" />
            <span
              style={{
                fontFamily: THEME.fonts.mono,
                fontSize: '12px',
                color: THEME.colors.statusVerified,
                backgroundColor: 'rgba(0,0,0,0.7)',
                padding: '4px 10px',
                borderRadius: '6px',
                fontWeight: 600,
              }}
            >
              EPISODE 000000 • 301 FRAMES (10.00s)
            </span>
          </div>

          {/* Bottom Video HUD Scrubber */}
          <div
            style={{
              position: 'absolute',
              bottom: 20,
              left: 20,
              right: 20,
              background: 'rgba(11, 15, 25, 0.9)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              padding: '12px 18px',
              borderRadius: '12px',
              fontFamily: THEME.fonts.mono,
              fontSize: '13px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <span style={{ color: THEME.colors.textMuted }}>CURRENT TIMESTAMP: </span>
              <span style={{ color: '#FFFFFF', fontWeight: 700 }}>
                {videoProgressSec.toFixed(2)}s / 10.00s
              </span>
            </div>
            <div style={{ color: THEME.colors.statusVerified, fontWeight: 700 }}>
              COVERAGE: COMPLETE_TASK
            </div>
          </div>
        </div>

        {/* Right: Actual Criteria Evaluation & Verification Verdict */}
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
              border: `1px solid ${THEME.colors.statusVerified}55`,
              borderRadius: '16px',
              padding: '20px',
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(17, 24, 39, 0.9) 100%)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted }}>
                  VERDICT ENGINE OUTCOME
                </div>
                <div style={{ fontSize: '32px', fontWeight: 900, color: THEME.colors.statusVerified }}>
                  VERIFIED
                </div>
              </div>
              <MetricBadge type="VERIFIED" label="COMPLETE TASK" size="lg" />
            </div>

            <p style={{ fontSize: '13px', color: THEME.colors.textSecondary, margin: '10px 0 0 0', lineHeight: 1.45 }}>
              All mandatory criteria are satisfied with verified spatio-temporal evidence across all 301 video frames.
            </p>
          </div>

          {/* Actual 4 Criteria from Report 03f2da9b */}
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
            <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted, marginBottom: '4px' }}>
              VERIFICATION CRITERIA BREAKDOWN
            </div>

            {[
              {
                id: 'object_transport_with_effector',
                name: 'Coordinated Transport',
                range: '2.10s – 8.47s (frames 63–254)',
                status: 'SATISFIED',
              },
              {
                id: 'object_reaches_target',
                name: 'Region Entry (Target)',
                range: '5.97s – 10.00s (frames 180–300)',
                status: 'SATISFIED',
              },
              {
                id: 'object_separates_from_effector',
                name: 'Effector Disengagement',
                range: '8.33s – 8.57s (frames 250–257)',
                status: 'SATISFIED',
              },
              {
                id: 'object_stable_in_target',
                name: 'Target Region Stability',
                range: '9.00s – 9.97s (frames 270–299)',
                status: 'SATISFIED',
              },
            ].map((crit) => (
              <div
                key={crit.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.06)',
                  borderRadius: '10px',
                  padding: '10px 14px',
                }}
              >
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#FFFFFF' }}>
                    {crit.name}
                  </div>
                  <div style={{ fontSize: '12px', fontFamily: THEME.fonts.mono, color: THEME.colors.textMuted, marginTop: '2px' }}>
                    {crit.range}
                  </div>
                </div>
                <MetricBadge type="SATISFIED" label="SATISFIED" size="sm" />
              </div>
            ))}
          </div>

          {/* Audit Link Note */}
          <div style={{ fontSize: '12px', color: THEME.colors.textMuted, fontFamily: THEME.fonts.mono }}>
            REPORT ID: 03f2da9b-6a11-4337-8b75-213c9728999f • 100% EVIDENCE TRACEABILITY
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene5} />
    </AbsoluteFill>
  );
};
