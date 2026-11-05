import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Badge } from '../components/Badge';
import { Img, staticFile } from 'remotion';

/**
 * Scene 9 – Conclusion.
 * Summarizes Boro value proposition with logo and call‑to‑action.
 */
const Scene9_Conclusion: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <Img
        src={staticFile('video-assets/brand/boro-mark.png')}
        style={{width: 180, height: 'auto', marginBottom: 20}}
      />
      <h1 style={{color: THEME.colors.brandOrange}}>
        Trust evidence. Automate with confidence.
      </h1>
      <p style={{color: THEME.colors.textPrimary, fontSize: 24, marginTop: 20}}>
        Boro makes robotic execution verifiable, reliable, and ready for industrial use.
      </p>
      <Badge type="verified" />
    </StageContainer>
  );
};
export default Scene9_Conclusion;
