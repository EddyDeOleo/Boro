import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { THEME } from '../constants/theme';

interface ArchitectureDiagramProps {
  activeStep?: number; // 1-5, or undefined for all
}

export const ArchitectureDiagram: React.FC<ArchitectureDiagramProps> = ({
  activeStep,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stages = [
    {
      num: '01',
      name: 'Meta Muse Spark',
      subtitle: 'Natural-Language Task Interpretation',
      badge: 'Meta AI',
      color: THEME.colors.brandCyanLight,
      desc: 'Translates unstructured prompts into formal TaskRoles and VerificationCriteria.',
    },
    {
      num: '02',
      name: 'Meta SAM 3.1',
      subtitle: 'Prompted Mask Segmentation & Tracking',
      badge: 'Meta AI',
      color: THEME.colors.brandCyan,
      desc: 'Initial bounding boxes expand into frame-by-frame 2D pixel-accurate contours.',
    },
    {
      num: '03',
      name: 'EventDetector',
      subtitle: 'Spatio-Temporal Event Extraction',
      badge: 'Boro Core',
      color: '#A78BFA',
      desc: 'Calculates continuous centroid displacement, motion similarity, and region overlaps.',
    },
    {
      num: '04',
      name: 'VerdictEngine',
      subtitle: 'Criteria & Coverage Verification',
      badge: 'Boro Core',
      color: THEME.colors.statusVerified,
      desc: 'Evaluates criteria against temporal events with explicit coverage scope verification.',
    },
    {
      num: '05',
      name: 'Execution Report',
      subtitle: 'Explainable Verdicts & Auditable Limitations',
      badge: 'Boro Output',
      color: THEME.colors.statusInsufficient,
      desc: 'Generates structured JSON reports detailing evidence, metrics, and sensor boundaries.',
    },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        gap: '20px',
        width: '100%',
        maxWidth: '1680px',
      }}
    >
      {stages.map((st, index) => {
        const stepNum = index + 1;
        const isCurrent = activeStep === undefined || activeStep === stepNum;
        const delay = index * 6;
        const entrySpring = spring({
          frame: frame - delay,
          fps,
          config: { damping: 14, mass: 0.8 },
        });

        const cardScale = isCurrent
          ? interpolate(entrySpring, [0, 1], [0.92, 1])
          : 0.96;
        const opacity = isCurrent
          ? interpolate(entrySpring, [0, 1], [0, 1])
          : 0.45;

        return (
          <div
            key={st.num}
            style={{
              backgroundColor: isCurrent ? THEME.colors.bgCard : '#0A0E17',
              border: `1px solid ${
                isCurrent ? st.color : THEME.colors.bgCardBorder
              }`,
              borderRadius: '16px',
              padding: '24px 20px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              boxShadow: isCurrent
                ? `0 10px 30px rgba(0, 0, 0, 0.4), 0 0 15px ${st.color}22`
                : 'none',
              transform: `scale(${cardScale})`,
              opacity,
              transition: 'border 0.2s ease',
            }}
          >
            <div>
              {/* Header row */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '14px',
                }}
              >
                <span
                  style={{
                    fontFamily: THEME.fonts.mono,
                    fontSize: '13px',
                    fontWeight: 700,
                    color: st.color,
                    letterSpacing: '0.08em',
                  }}
                >
                  STAGE {st.num}
                </span>
                <span
                  style={{
                    fontSize: '11px',
                    padding: '3px 8px',
                    borderRadius: '9999px',
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    color: THEME.colors.textSecondary,
                    fontWeight: 600,
                  }}
                >
                  {st.badge}
                </span>
              </div>

              {/* Title */}
              <div
                style={{
                  fontSize: '20px',
                  fontWeight: 700,
                  color: '#FFFFFF',
                  marginBottom: '6px',
                  letterSpacing: '-0.01em',
                }}
              >
                {st.name}
              </div>

              <div
                style={{
                  fontSize: '13px',
                  fontWeight: 600,
                  color: st.color,
                  marginBottom: '14px',
                  lineHeight: 1.35,
                }}
              >
                {st.subtitle}
              </div>
            </div>

            {/* Description */}
            <div
              style={{
                fontSize: '13px',
                color: THEME.colors.textSecondary,
                lineHeight: 1.5,
                borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                paddingTop: '14px',
              }}
            >
              {st.desc}
            </div>
          </div>
        );
      })}
    </div>
  );
};
