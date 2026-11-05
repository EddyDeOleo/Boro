import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { TimelineChart } from '../components/TimelineChart';
import { Badge } from '../components/Badge';
import { Video } from 'remotion';

/**
 * Scene 5 – Temporal verification.
 * Shows the event timeline derived from Boro's analysis of the critical episode.
 */
const Scene5_TemporalVerification: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        Temporal verification – event analysis
      </h2>
      <p style={{color: THEME.colors.textSecondary, marginBottom: 10}}>
        Boro extracts discrete events from the robot footage to reason about task completion.
      </p>
      <TimelineChart />
    </StageContainer>
  );
};
export default Scene5_TemporalVerification;
