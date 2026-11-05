import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Video, staticFile } from 'remotion';

/**
 * Scene 1 – Problem (slip scenario).
 * Shows the critical episode clip with a text overlay.
 */
const Scene1_Problem: React.FC<{start: number; duration: number}> = ({start, duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{marginBottom: 20, color: THEME.colors.textPrimary}}>
        Problem: Apparent success but the object slips
      </h2>
      <Video
        src={staticFile('videos/episode_000025_critical.mp4')}
        startFrom={0}
        endAt={51}
        style={{width: '100%', height: 'auto'}}
      />
    </StageContainer>
  );
};
export default Scene1_Problem;
