import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Img, staticFile } from 'remotion';

/**
 * Scene 4 – Visual perception (SAM segmentation & tracking).
 * Shows a representative overlay image.
 */
const Scene4_VisualPerception: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        Visual perception – SAM 3.1 segmentation & tracking
      </h2>
      <Img
        src={staticFile('evaluation/dual_mask_visualization/frame_20_overlay.jpg')}
        style={{width: '100%', maxHeight: '70vh', objectFit: 'contain', marginTop: 20}}
      />
    </StageContainer>
  );
};
export default Scene4_VisualPerception;
