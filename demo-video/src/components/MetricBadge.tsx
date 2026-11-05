import React from 'react';
import { THEME } from '../constants/theme';

export interface MetricBadgeProps {
  type: 'VERIFIED' | 'INSUFFICIENT_EVIDENCE' | 'FAILED' | 'SATISFIED' | 'MANDATORY' | 'INFO';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const MetricBadge: React.FC<MetricBadgeProps> = ({
  type,
  label,
  size = 'md',
}) => {
  let bg = 'rgba(255, 255, 255, 0.08)';
  let border = 'rgba(255, 255, 255, 0.15)';
  let color = THEME.colors.textPrimary;
  let text = label || type;

  switch (type) {
    case 'VERIFIED':
    case 'SATISFIED':
      bg = 'rgba(16, 185, 129, 0.15)';
      border = 'rgba(16, 185, 129, 0.4)';
      color = THEME.colors.statusVerified;
      break;
    case 'INSUFFICIENT_EVIDENCE':
      bg = 'rgba(245, 158, 11, 0.15)';
      border = 'rgba(245, 158, 11, 0.4)';
      color = THEME.colors.statusInsufficient;
      text = label || 'INSUFFICIENT EVIDENCE';
      break;
    case 'FAILED':
      bg = 'rgba(239, 68, 68, 0.15)';
      border = 'rgba(239, 68, 68, 0.4)';
      color = THEME.colors.statusFailed;
      break;
    case 'MANDATORY':
      bg = 'rgba(6, 104, 225, 0.15)';
      border = 'rgba(6, 104, 225, 0.4)';
      color = THEME.colors.brandCyanLight;
      break;
    case 'INFO':
      bg = 'rgba(0, 210, 255, 0.12)';
      border = 'rgba(0, 210, 255, 0.35)';
      color = THEME.colors.brandCyan;
      break;
  }

  const padding =
    size === 'sm' ? '4px 10px' : size === 'lg' ? '10px 22px' : '6px 14px';
  const fontSize = size === 'sm' ? '12px' : size === 'lg' ? '18px' : '14px';

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        backgroundColor: bg,
        border: `1px solid ${border}`,
        borderRadius: '9999px',
        padding,
        color,
        fontSize,
        fontFamily: THEME.fonts.mono,
        fontWeight: 700,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
      }}
    >
      <div
        style={{
          width: size === 'sm' ? '6px' : '8px',
          height: size === 'sm' ? '6px' : '8px',
          borderRadius: '50%',
          backgroundColor: color,
        }}
      />
      {text}
    </div>
  );
};
