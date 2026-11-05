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
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';

export const Scene8_Conclusion: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.8 },
  });

  const mottoEntrance = spring({
    frame: frame - 120, // Reveal motto prominently after narration sets up
    fps,
    config: { damping: 12, mass: 0.7 },
  });

  const glowPulse = Math.sin(frame / 10) * 0.2 + 0.8;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: THEME.colors.bgDark,
        fontFamily: THEME.fonts.heading,
        color: THEME.colors.textPrimary,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: '80px',
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('audio/scene8.wav')} />
      <Header sceneNumber="08" sceneTitle="Conclusion & Mission" />

      {/* Background ambient lighting */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '900px',
          height: '600px',
          background: 'radial-gradient(ellipse at center, rgba(6, 104, 225, 0.25) 0%, rgba(11, 15, 25, 0) 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Main Content */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '24px',
          opacity: entrance,
          transform: `scale(${interpolate(entrance, [0, 1], [0.92, 1])})`,
          zIndex: 10,
        }}
      >
        {/* Brand Mark with Pulse Glow */}
        <div style={{ position: 'relative' }}>
          <div
            style={{
              position: 'absolute',
              inset: -15,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(0, 210, 255, 0.4) 0%, transparent 70%)',
              filter: 'blur(15px)',
              transform: `scale(${glowPulse})`,
            }}
          />
          <Img
            src={staticFile('brand/boro-mark.png')}
            style={{
              position: 'relative',
              width: '110px',
              height: '110px',
              objectFit: 'contain',
              filter: 'drop-shadow(0 0 30px rgba(6, 104, 225, 0.8))',
            }}
          />
        </div>

        <div>
          <div
            style={{
              fontSize: '64px',
              fontWeight: 900,
              letterSpacing: '-0.03em',
              lineHeight: 1,
              color: '#FFFFFF',
            }}
          >
            BORO
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: 700,
              letterSpacing: '0.12em',
              color: THEME.colors.brandCyan,
              textTransform: 'uppercase',
              marginTop: '8px',
            }}
          >
            Robotic Execution Intelligence
          </div>
        </div>

        {/* Tech Stack Callout */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: `1px solid ${THEME.colors.bgCardBorder}`,
            padding: '10px 24px',
            borderRadius: '9999px',
            fontFamily: THEME.fonts.mono,
            fontSize: '14px',
            color: THEME.colors.textSecondary,
          }}
        >
          <span style={{ color: THEME.colors.brandCyanLight, fontWeight: 700 }}>Meta Muse Spark</span>
          <span>•</span>
          <span style={{ color: THEME.colors.brandCyan, fontWeight: 700 }}>Meta SAM 3.1</span>
          <span>•</span>
          <span style={{ color: THEME.colors.statusVerified, fontWeight: 700 }}>Boro VerdictEngine</span>
        </div>

        {/* Closing Climax Motto */}
        <div
          style={{
            marginTop: '20px',
            opacity: mottoEntrance,
            transform: `translateY(${interpolate(mottoEntrance, [0, 1], [25, 0])}px)`,
            maxWidth: '1100px',
          }}
        >
          <div
            style={{
              fontSize: '44px',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              lineHeight: 1.25,
              background: 'linear-gradient(180deg, #FFFFFF 20%, #CBD5E1 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
            }}
          >
            &ldquo;A robot&rsquo;s movement is not proof of success. Evidence is.&rdquo;
          </div>
        </div>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene8} />
    </AbsoluteFill>
  );
};
