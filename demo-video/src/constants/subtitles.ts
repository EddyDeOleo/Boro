export interface SubtitleCue {
  startFrame: number;
  endFrame: number;
  text: string;
}

export const SUBTITLES: Record<string, SubtitleCue[]> = {
  scene1: [
    {
      startFrame: 30,
      endFrame: 165,
      text: 'In autonomous robotics, movement is often mistaken for success.',
    },
    {
      startFrame: 165,
      endFrame: 340,
      text: 'A robot reaches its target coordinates, but did the object stay in the gripper? Or did it slip unnoticed?',
    },
    {
      startFrame: 340,
      endFrame: 490,
      text: 'Blindly trusting coordinate execution hides critical failures.',
    },
  ],
  scene2: [
    {
      startFrame: 30,
      endFrame: 210,
      text: 'Introducing Boro: a robotic execution intelligence platform built for the Meta Global AI Developer Hackathon.',
    },
    {
      startFrame: 210,
      endFrame: 480,
      text: 'Boro bridges the gap between natural language instructions and physical reality, converting visual streams into auditable verification evidence.',
    },
  ],
  scene3: [
    {
      startFrame: 30,
      endFrame: 240,
      text: 'The pipeline begins with Meta Muse Spark. Rather than relying on rigid rules...',
    },
    {
      startFrame: 240,
      endFrame: 530,
      text: '...Muse Spark interprets natural language instructions into structured task definitions, assigning semantic roles and establishing criteria with explicit mandatory requirements.',
    },
  ],
  scene4: [
    {
      startFrame: 30,
      endFrame: 250,
      text: 'Next, Meta SAM 3.1 performs spatio-temporal segmentation. From initial bounding boxes, SAM tracks the gripper and object across every frame.',
    },
    {
      startFrame: 250,
      endFrame: 560,
      text: 'Boro extracts two-dimensional centroid trajectories, providing visual evidence without assuming unmeasured physical forces.',
    },
  ],
  scene5: [
    {
      startFrame: 30,
      endFrame: 270,
      text: 'Here is episode zero. The robot cleanly grasps the cube, transports it across the table, releases it inside the container, and retracts.',
    },
    {
      startFrame: 270,
      endFrame: 540,
      text: 'Every mandatory criterion is satisfied across all three hundred and one frames, earning an explainable verdict of Verified.',
    },
  ],
  scene6: [
    {
      startFrame: 30,
      endFrame: 280,
      text: 'Now examine episode twenty-five. In this critical 1.67-second segment, the cube slips prematurely.',
    },
    {
      startFrame: 280,
      endFrame: 600,
      text: 'Boro detects relative separation, but because the clip ends before confirming placement, VerdictEngine issues an explainable verdict: Insufficient Evidence.',
    },
  ],
  scene7: [
    {
      startFrame: 30,
      endFrame: 270,
      text: "Through Boro's modern interface, roboticists inspect temporal timelines, review confidence scores, and read transparent limitations.",
    },
    {
      startFrame: 270,
      endFrame: 560,
      text: 'Boro explicitly documents boundary assumptions, such as two-dimensional constraints, ensuring human operators can audit every verification.',
    },
  ],
  scene8: [
    {
      startFrame: 30,
      endFrame: 220,
      text: 'Built for the Meta Global AI Developer Hackathon 2026, Boro delivers honest, auditable intelligence to robotic workflows.',
    },
    {
      startFrame: 220,
      endFrame: 410,
      text: "Because a robot's movement is not proof of success. Evidence is.",
    },
  ],
};
