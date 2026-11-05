import React from 'react';
import { THEME } from '../../constants/theme';
import { Sequence } from 'remotion';

// A simple container that provides a dark panel background and padding.
// Used to wrap each narrative stage.
export const StageContainer: React.FC<{
  children: React.ReactNode;
  // start frame for the stage within the overall composition
  start: number;
  // duration in frames
  duration: number;
}> = ({ children, start, duration }) => {
  return (
    <Sequence from={start} durationInFrames={duration}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: THEME.colors.panelBg,
          color: THEME.colors.textPrimary,
          padding: 40,
          fontFamily: THEME.fonts.heading,
        }}
      >
        {children}
      </div>
    </Sequence>
  );
};
