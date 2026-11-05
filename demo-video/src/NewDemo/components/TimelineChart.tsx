import React, {useEffect, useState} from 'react';
import {THEME} from '../../constants/theme';
import { Img, staticFile } from 'remotion';

/**
 * TimelineChart – simple SVG timeline based on Boro event data.
 * It loads the JSONL file at runtime via staticFile and parses timestamps.
 * For simplicity we draw a horizontal line with markers.
 */
export const TimelineChart: React.FC = () => {
  const [events, setEvents] = useState<number[]>([]);

  useEffect(() => {
    // Load the events JSONL file as text.
    const load = async () => {
      const resp = await fetch(staticFile('evaluation/sam_episode_000025_events.jsonl'));
      const text = await resp.text();
      // Each line is a JSON object with a "timestamp" field (seconds).
      const lines = text.split(/\r?\n/).filter(l => l.trim());
      const ts = lines.map(l => {
        try { return JSON.parse(l).timestamp; } catch { return null; }
      }).filter((t): t is number => typeof t === 'number');
      setEvents(ts);
    };
    load();
  }, []);

  const width = 800;
  const height = 120;
  const padding = 20;
  const maxTs = Math.max(...events, 0);
  const scale = maxTs ? (width - 2 * padding) / maxTs : 0;

  return (
    <svg width={width} height={height} style={{backgroundColor: THEME.colors.panelBg}}>
      {/* baseline */}
      <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke={THEME.colors.timelineBar} strokeWidth={2} />
      {/* markers */}
      {events.map((t, i) => (
        <g key={i}>
          <line
            x1={padding + t * scale}
            y1={height / 2 - 10}
            x2={padding + t * scale}
            y2={height / 2 + 10}
            stroke={THEME.colors.techTeal}
            strokeWidth={2}
          />
          <text
            x={padding + t * scale}
            y={height / 2 - 15}
            fill={THEME.colors.textSecondary}
            fontSize={12}
            fontFamily={THEME.fonts.heading}
            textAnchor="middle"
          >
            {t.toFixed(1)}s
          </text>
        </g>
      ))}
    </svg>
  );
};
