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
import { ArchitectureDiagram } from '../components/ArchitectureDiagram';
import { SUBTITLES } from '../constants/subtitles';
import { THEME } from '../constants/theme';

export const Scene2_Solution: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const logoSpring = spring({
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
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '120px 60px 100px 60px',
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('audio/scene2.wav')} />
      <Header sceneNumber="02" sceneTitle="Introducing Boro" />

      {/* Hero Section */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
          gap: '16px',
          opacity: logoSpring,
          transform: `scale(${interpolate(logoSpring, [0, 1], [0.94, 1])})`,
          zIndex: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '22px' }}>
          <Img
            src={staticFile('brand/boro-mark.png')}
            style={{
              width: '80px',
              height: '80px',
              objectFit: 'contain',
              filter: 'drop-shadow(0 0 24px rgba(6, 104, 225, 0.6))',
            }}
          />
          <div style={{ textAlign: 'left' }}>
            <div
              style={{
                fontSize: '52px',
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
                fontSize: '16px',
                fontWeight: 700,
                letterSpacing: '0.12em',
                color: THEME.colors.brandCyan,
                textTransform: 'uppercase',
                marginTop: '4px',
              }}
            >
              Robotic Execution Intelligence
            </div>
          </div>
        </div>

        <p
          style={{
            fontSize: '22px',
            color: THEME.colors.textSecondary,
            maxWidth: '920px',
            lineHeight: 1.45,
            margin: '8px 0 0 0',
          }}
        >
          Grounding natural-language task specifications in verifiable spatio-temporal video evidence
          powered by Meta AI foundational models.
        </p>
      </div>

      {/* 5-Stage Architecture Diagram */}
      <div style={{ width: '100%', display: 'flex', justifyContent: 'center', zIndex: 10 }}>
        <ArchitectureDiagram />
      </div>

      {/* Bottom Principle Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '24px',
          background: 'rgba(255, 255, 255, 0.04)',
          border: `1px solid ${THEME.colors.bgCardBorder}`,
          padding: '12px 28px',
          borderRadius: '9999px',
          fontFamily: THEME.fonts.mono,
          fontSize: '14px',
          zIndex: 10,
        }}
      >
        <span style={{ color: THEME.colors.textMuted }}>CORE PRINCIPLE:</span>
        <span style={{ color: THEME.colors.statusVerified, fontWeight: 700 }}>EVIDENCE BEFORE CONCLUSIONS</span>
        <span style={{ color: THEME.colors.textMuted }}>•</span>
        <span style={{ color: THEME.colors.brandCyanLight, fontWeight: 600 }}>META FOUNDATIONAL MODEL INTEGRATION</span>
      </div>

      <SubtitleOverlay cues={SUBTITLES.scene2} />
    </AbsoluteFill>
  );
};
