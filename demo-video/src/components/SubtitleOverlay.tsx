import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';
import { SubtitleCue } from '../constants/subtitles';
import { THEME } from '../constants/theme';

interface SubtitleOverlayProps {
  cues: SubtitleCue[];
}

export const SubtitleOverlay: React.FC<SubtitleOverlayProps> = ({ cues }) => {
  const frame = useCurrentFrame();

  const activeCue = cues.find(
    (c) => frame >= c.startFrame && frame <= c.endFrame
  );

  if (!activeCue) {
    return null;
  }

  // Smooth fade-in and fade-out near cue boundaries
  const fadeIn = interpolate(
    frame,
    [activeCue.startFrame, activeCue.startFrame + 8],
    [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const fadeOut = interpolate(
    frame,
    [activeCue.endFrame - 8, activeCue.endFrame],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const opacity = Math.min(fadeIn, fadeOut);

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 50,
        left: '50%',
        transform: 'translateX(-50%)',
        maxWidth: '1280px',
        width: '90%',
        display: 'flex',
        justifyContent: 'center',
        zIndex: 60,
        opacity,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          background: 'rgba(11, 15, 25, 0.88)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          boxShadow: '0 12px 36px rgba(0, 0, 0, 0.65)',
          padding: '16px 36px',
          borderRadius: '16px',
          textAlign: 'center',
        }}
      >
        <span
          style={{
            fontSize: '24px',
            lineHeight: 1.45,
            fontWeight: 600,
            color: '#FFFFFF',
            fontFamily: THEME.fonts.heading,
            letterSpacing: '-0.01em',
            textShadow: '0 2px 8px rgba(0,0,0,0.8)',
          }}
        >
          {activeCue.text}
        </span>
      </div>
    </div>
  );
};
