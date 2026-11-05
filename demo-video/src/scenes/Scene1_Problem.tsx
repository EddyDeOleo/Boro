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
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';
import { MetricBadge } from '../components/MetricBadge';

export const Scene1_Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Intro animation
  const titleAnim = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.8 },
  });

  // Freeze/glitch effect around frame 200 (approx when cube drops)
  const isCriticalWindow = frame >= 180;
  const pulseAlert = Math.sin(frame / 4) * 0.5 + 0.5;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: THEME.colors.bgDark,
        fontFamily: THEME.fonts.heading,
        color: THEME.colors.textPrimary,
        overflow: 'hidden',
      }}
    >
      {/* TODO: replace with natural narration audio */}
      <Audio src={staticFile('audio/scene1.wav')} />
      <Header sceneNumber="01" sceneTitle="The Verification Problem" />

      {/* Main Content Layout */}
      <div
        style={{
          position: 'absolute',
          top: 125,
          left: 60,
          right: 60,
          bottom: 145,
          display: 'grid',
          gridTemplateColumns: '1.2fr 0.8fr',
          gap: '40px',
          alignItems: 'center',
        }}
      >
        {/* Left: Video Monitor Box */}
        <div
          style={{
            position: 'relative',
            height: '100%',
            backgroundColor: '#000000',
            borderRadius: '20px',
            overflow: 'hidden',
            border: isCriticalWindow
              ? '2px solid rgba(239, 68, 68, 0.7)'
              : `1px solid ${THEME.colors.bgCardBorder}`,
            boxShadow: isCriticalWindow
              ? '0 0 40px rgba(239, 68, 68, 0.25)'
              : '0 20px 50px rgba(0, 0, 0, 0.7)',
          }}
        >
          {/* Looping critical robot video */}
          <Video
            src={staticFile('videos/episode_000025_critical.mp4')}
            playbackRate={0.65}
            loop
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              transform: `scale(${interpolate(frame, [0, 540], [1, 1.05])})`,
            }}
          />

          {/* HUD Overlays */}
          <div
            style={{
              position: 'absolute',
              top: 20,
              left: 20,
              display: 'flex',
              gap: '10px',
            }}
          >
            <MetricBadge type="FAILED" label="REAL ROBOT FOOTAGE" size="sm" />
            <span
              style={{
                fontFamily: THEME.fonts.mono,
                fontSize: '12px',
                color: '#EF4444',
                backgroundColor: 'rgba(0,0,0,0.7)',
                padding: '4px 10px',
                borderRadius: '6px',
                fontWeight: 600,
              }}
            >
              EPISODE 000025 • CRITICAL SLIP WINDOW
            </span>
          </div>

          {/* Critical diagnostic callout at slip point */}
          {isCriticalWindow && (
            <div
              style={{
                position: 'absolute',
                bottom: 24,
                left: 24,
                right: 24,
                background: 'rgba(15, 23, 42, 0.92)',
                backdropFilter: 'blur(10px)',
                border: '1px solid #EF4444',
                padding: '16px 20px',
                borderRadius: '12px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                boxShadow: `0 0 20px rgba(239, 68, 68, ${pulseAlert * 0.4})`,
              }}
            >
              <div>
                <div style={{ color: '#EF4444', fontWeight: 700, fontSize: '13px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Critical Execution Anomaly Detected
                </div>
                <div style={{ color: THEME.colors.textSecondary, fontSize: '12px', marginTop: '4px' }}>
                  Effector separated from target object prematurely before entering target zone.
                </div>
              </div>
              <MetricBadge type="FAILED" label="SILENT FAILURE" size="sm" />
            </div>
          )}
        </div>

        {/* Right: Analytical Narrative & Challenge Cards */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
            opacity: titleAnim,
            transform: `translateY(${interpolate(titleAnim, [0, 1], [30, 0])}px)`,
          }}
        >
          <div>
            <div
              style={{
                color: THEME.colors.brandCyan,
                fontSize: '14px',
                fontFamily: THEME.fonts.mono,
                fontWeight: 700,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                marginBottom: '10px',
              }}
            >
              The Autonomous Blindspot
            </div>
            <h1
              style={{
                fontSize: '44px',
                lineHeight: 1.15,
                fontWeight: 800,
                letterSpacing: '-0.02em',
                margin: 0,
              }}
            >
              Movement is Not Proof of Execution.
            </h1>
          </div>

          <p
            style={{
              fontSize: '18px',
              color: THEME.colors.textSecondary,
              lineHeight: 1.55,
              margin: 0,
            }}
          >
            Robotic control stacks report success when motors hit target coordinates.
            Without empirical vision-based verification, drops, slips, and partial executions
            pass as completed tasks.
          </p>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              paddingTop: '20px',
            }}
          >
            {[
              {
                icon: '✖',
                title: 'Hallucinated Task Success',
                desc: 'Robot arm reaches target bin; dropped object is left behind on table.',
              },
              {
                icon: '✖',
                title: 'Sensorless Blindspots',
                desc: 'No tactile feedback or load cells to verify physical grip stability.',
              },
              {
                icon: '✖',
                title: 'Uncalibrated Guessing',
                desc: 'Naive end-state models declare success if objects happen to tumble in.',
              },
            ].map((item, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.06)',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                }}
              >
                <span style={{ color: '#EF4444', fontWeight: 800, fontSize: '15px' }}>
                  {item.icon}
                </span>
                <div>
                  <div style={{ fontSize: '15px', fontWeight: 700, color: '#FFFFFF' }}>
                    {item.title}
                  </div>
                  <div style={{ fontSize: '13px', color: THEME.colors.textSecondary, marginTop: '2px' }}>
                    {item.desc}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene1} />
    </AbsoluteFill>
  );
};
