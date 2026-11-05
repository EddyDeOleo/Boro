import React from 'react';
import { Img, staticFile } from 'remotion';
import { THEME } from '../constants/theme';

interface HeaderProps {
  sceneNumber: string;
  sceneTitle: string;
}

export const Header: React.FC<HeaderProps> = ({ sceneNumber, sceneTitle }) => {
  return (
    <div
      style={{
        position: 'absolute',
        top: 40,
        left: 60,
        right: 60,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 50,
      }}
    >
      {/* Brand & Scene Tag */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <Img
          src={staticFile('brand/boro-mark.png')}
          style={{ width: '44px', height: '44px', objectFit: 'contain' }}
        />
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '22px', fontWeight: 800, letterSpacing: '-0.02em', color: '#FFFFFF' }}>
              BORO
            </span>
            <span style={{ fontSize: '12px', color: THEME.colors.textMuted }}>•</span>
            <span style={{ fontSize: '13px', color: THEME.colors.brandCyan, fontFamily: THEME.fonts.mono, fontWeight: 700 }}>
              {sceneNumber}
            </span>
          </div>
          <span style={{ fontSize: '13px', color: THEME.colors.textSecondary, letterSpacing: '0.04em', textTransform: 'uppercase', fontWeight: 600 }}>
            {sceneTitle}
          </span>
        </div>
      </div>

      {/* Meta Hackathon Badge */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(17, 24, 39, 0.75)',
          backdropFilter: 'blur(12px)',
          border: `1px solid ${THEME.colors.bgCardBorder}`,
          padding: '8px 16px',
          borderRadius: '9999px',
          fontSize: '13px',
          fontWeight: 600,
        }}
      >
        <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: THEME.colors.brandCyanLight }} />
        <span style={{ color: THEME.colors.brandCyanLight, fontWeight: 700 }}>Meta</span>
        <span style={{ color: THEME.colors.textSecondary }}>Global AI Developer Hackathon 2026</span>
      </div>
    </div>
  );
};
