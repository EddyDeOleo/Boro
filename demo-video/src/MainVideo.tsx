import React from 'react';
import { AbsoluteFill, Series } from 'remotion';
import { Scene1_Problem } from './scenes/Scene1_Problem';
import { Scene2_Solution } from './scenes/Scene2_Solution';
import { Scene3_MuseSpark } from './scenes/Scene3_MuseSpark';
import { Scene4_SAMTracking } from './scenes/Scene4_SAMTracking';
import { Scene5_VerifiedEpisode } from './scenes/Scene5_VerifiedEpisode';
import { Scene6_InsufficientEpisode } from './scenes/Scene6_InsufficientEpisode';
import { Scene7_FrontendPlatform } from './scenes/Scene7_FrontendPlatform';
import { Scene8_Conclusion } from './scenes/Scene8_Conclusion';

export const MainVideo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: '#0B0F19' }}>
      <Series>
        {/* Scene 1: 00:00 - 00:18 (540 frames) */}
        <Series.Sequence durationInFrames={540}>
          <Scene1_Problem />
        </Series.Sequence>

        {/* Scene 2: 00:18 - 00:40 (660 frames) */}
        <Series.Sequence durationInFrames={660}>
          <Scene2_Solution />
        </Series.Sequence>

        {/* Scene 3: 00:40 - 01:05 (750 frames) */}
        <Series.Sequence durationInFrames={750}>
          <Scene3_MuseSpark />
        </Series.Sequence>

        {/* Scene 4: 01:05 - 01:30 (750 frames) */}
        <Series.Sequence durationInFrames={750}>
          <Scene4_SAMTracking />
        </Series.Sequence>

        {/* Scene 5: 01:30 - 01:55 (750 frames) */}
        <Series.Sequence durationInFrames={750}>
          <Scene5_VerifiedEpisode />
        </Series.Sequence>

        {/* Scene 6: 01:55 - 02:22 (810 frames) */}
        <Series.Sequence durationInFrames={810}>
          <Scene6_InsufficientEpisode />
        </Series.Sequence>

        {/* Scene 7: 02:22 - 02:45 (690 frames) */}
        <Series.Sequence durationInFrames={690}>
          <Scene7_FrontendPlatform />
        </Series.Sequence>

        {/* Scene 8: 02:45 - 03:00 (450 frames) */}
        <Series.Sequence durationInFrames={450}>
          <Scene8_Conclusion />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
