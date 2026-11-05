import React from 'react';
import { THEME } from '../constants/theme';
import { StageContainer } from './components/StageContainer';
import Scene1_Problem from './scenes/Scene1_Problem';
import Scene2_Intro from './scenes/Scene2_Intro';
import Scene3_TaskInterpretation from './scenes/Scene3_TaskInterpretation';
import Scene4_VisualPerception from './scenes/Scene4_VisualPerception';
import Scene5_TemporalVerification from './scenes/Scene5_TemporalVerification';
import Scene6_SuccessfulExecution from './scenes/Scene6_SuccessfulExecution';
import Scene7_AmbiguousExecution from './scenes/Scene7_AmbiguousExecution';
import Scene8_ProductExperience from './scenes/Scene8_ProductExperience';
import Scene9_Conclusion from './scenes/Scene9_Conclusion';

/**
 * NewDemo video composition – isolated from the original demo.
 * It assembles nine stages, each wrapped in a StageContainer.
 * The composition length matches the existing theme video length (180 s, 5400 frames).
 */
export const NewDemoVideo: React.FC = () => {
  // Allocate roughly equal duration to each stage (≈600 frames ≈10 s).
  const stageDuration = Math.floor(THEME.video.totalFrames / 9);
  const stages = [
    { component: Scene1_Problem, label: 'Problem' },
    { component: Scene2_Intro, label: 'Intro' },
    { component: Scene3_TaskInterpretation, label: 'TaskInterpretation' },
    { component: Scene4_VisualPerception, label: 'VisualPerception' },
    { component: Scene5_TemporalVerification, label: 'TemporalVerification' },
    { component: Scene6_SuccessfulExecution, label: 'SuccessfulExecution' },
    { component: Scene7_AmbiguousExecution, label: 'AmbiguousExecution' },
    { component: Scene8_ProductExperience, label: 'ProductExperience' },
    { component: Scene9_Conclusion, label: 'Conclusion' },
  ];

  return (
    <>
      {stages.map((s, idx) => {
        const Start = idx * stageDuration;
        const Component = s.component;
        return (
          <Component key={s.label} start={Start} duration={stageDuration} />
        );
      })}
    </>
  );
};
