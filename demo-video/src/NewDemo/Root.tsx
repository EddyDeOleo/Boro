import React from 'react';
import { Composition } from 'remotion';
import { NewDemoVideo } from './Video';
import { THEME } from '../constants/theme';

/**
 * Isolated root for the new Boro demo composition.
 * This file allows rendering the NewDemoVideo without altering the original Root.
 */
export const NewDemoRoot: React.FC = () => {
  return (
    <Composition
      id="BoroNewDemo"
      component={NewDemoVideo}
      width={THEME.video.width}
      height={THEME.video.height}
      fps={THEME.video.fps}
      durationInFrames={THEME.video.totalFrames}
    />
  );
};
