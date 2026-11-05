import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Img, Video, staticFile } from 'remotion';

/**
 * Scene 2 – Introduction to Boro.
 * Shows the brand logo and a tagline.
 */
const Scene2_Intro: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <Img
        src={staticFile('video-assets/brand/boro-mark.png')}
        style={{width: 200, height: 'auto', marginBottom: 20}}
      />
      <h1 style={{color: THEME.colors.brandOrange}}>
        Boro – Robotic Execution Intelligence
      </h1>
      <p style={{color: THEME.colors.textPrimary, fontSize: 24}}>
        Define what should happen. See what actually happened. Verify the evidence.
      </p>
    </StageContainer>
  );
};
export default Scene2_Intro;
