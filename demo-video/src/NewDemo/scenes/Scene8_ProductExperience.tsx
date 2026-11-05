import React from 'react';
import { THEME } from '../../constants/theme';
import { StageContainer } from '../components/StageContainer';
import { Video, staticFile } from 'remotion';

/**
 * Scene 8 – Product experience (frontend UI).
 * Shows the real Boro frontend demo video.
 */
const Scene8_ProductExperience: React.FC<{start:number;duration:number}> = ({start,duration}) => {
  return (
    <StageContainer start={start} duration={duration}>
      <h2 style={{color: THEME.colors.textPrimary}}>
        Boro platform – real UI walkthrough
      </h2>
      <Video
        src={staticFile('video-assets/frontend/boro_frontend_demo.mp4')}
        startFrom={0}
        endAt={15}
        style={{marginTop: 20, width: '100%', height: 'auto'}}
      />
    </StageContainer>
  );
};
export default Scene8_ProductExperience;
