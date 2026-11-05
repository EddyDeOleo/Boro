import React from 'react';
import { Composition } from 'remotion';
import { MainVideo } from './MainVideo';
import { THEME } from './constants/theme';

export const Root: React.FC = () => {
  return (
    <Composition
      id="BoroDemo"
      component={MainVideo}
      durationInFrames={THEME.video.totalFrames}
      fps={THEME.video.fps}
      width={THEME.video.width}
      height={THEME.video.height}
      defaultProps={{}}
    />
  );
};
