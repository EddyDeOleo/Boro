import React from 'react';
import { THEME } from '../../constants/theme';

/**
 * Badge component for status indicators.
 * `type` determines background colour defined in THEME.
 */
export const Badge: React.FC<{type: 'verified' | 'insufficient' | 'failed'}> = ({type}) => {
  const colors = {
    verified: THEME.colors.statusVerified,
    insufficient: THEME.colors.statusInsufficient,
    failed: THEME.colors.statusFailed,
  };
  const labels = {
    verified: 'VERIFIED',
    insufficient: 'INSUFFICIENT EVIDENCE',
    failed: 'FAILED',
  };
  return (
    <div
      style={{
        backgroundColor: colors[type],
        color: '#fff',
        padding: '4px 12px',
        borderRadius: 4,
        fontFamily: THEME.fonts.heading,
        fontSize: 18,
        display: 'inline-block',
      }}
    >
      {labels[type]}
    </div>
  );
};
