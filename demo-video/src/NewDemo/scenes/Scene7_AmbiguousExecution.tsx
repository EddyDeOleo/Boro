import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Badge } from '../components/Badge';
import { Video, staticFile } from 'remotion';

/**
 * Scene 7 – Ambiguous execution (INSUFFICIENT EVIDENCE).
 * Plays the critical episode segment and shows an INSUFFICIENT EVIDENCE badge.
 */
const Scene7_AmbiguousExecution: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        Ambiguous execution – INSUFFICIENT EVIDENCE
      </h2>
      <Badge type="insufficient" />
      <Video
        src={staticFile('videos/episode_000025_critical.mp4')}
        startFrom={5}
        endAt={15}
        style={{marginTop: 20, width: '100%', height: 'auto'}}
      />
    </StageContainer>
  );
};
export default Scene7_AmbiguousExecution;
