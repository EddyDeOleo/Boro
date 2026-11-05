import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Badge } from '../components/Badge';
import { Video, staticFile } from 'remotion';

/**
 * Scene 6 — Successful execution (VERIFIED).
 * Plays the verified episode and shows a VERIFIED badge.
 */
const Scene6_SuccessfulExecution: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        Verified execution
      </h2>
      <Badge type="verified" />
      <Video
        src={staticFile('videos/episode_000000.mp4')}
        startFrom={0}
        endAt={300}
        style={{marginTop: 20, width: '100%', height: 'auto'}}
      />
    </StageContainer>
  );
};

export default Scene6_SuccessfulExecution;
