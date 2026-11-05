import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Img, Video, staticFile } from 'remotion';

/**
 * Scene 3 – Task interpretation (Muse Spark).
 * Displays a simple representation of the natural‑language task becoming a structured goal.
 */
const Scene3_TaskInterpretation: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        How the user instruction is parsed
      </h2>
      <p style={{color: THEME.colors.textSecondary, fontSize: 20}}>
        "Pick the red cube and place it on the green platform"
      </p>
      <Img
        src={staticFile('video-assets/brand/boro-mark.png')}
        style={{width: 80, height: 'auto', opacity: 0.2, position: 'absolute', top: 10, right: 10}}
      />
      <p style={{color: THEME.colors.textSecondary, marginTop: 30}}>
        → Structured task criteria (object type, target location, constraints)
      </p>
    </StageContainer>
  );
};
export default Scene3_TaskInterpretation;
