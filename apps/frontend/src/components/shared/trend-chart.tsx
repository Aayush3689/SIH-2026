'use client';

import { useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '@/lib/utils';

export interface TrendSeries {
  id: string;
  label: string;
  color: string;
  values: number[];
}

export function TrendChart({
  title,
  subtitle,
  series,
  className,
  height = 170,
}: {
  title: string;
  subtitle?: string;
  series: TrendSeries[];
  className?: string;
  height?: number;
}) {
  const reduceMotion = useReducedMotion();
  const width = 500;
  const left = 10;
  const right = 10;
  const top = 12;
  const bottom = 20;
  const validValues = series.flatMap((item) => item.values).filter((value) => Number.isFinite(value));
  const min = validValues.length ? Math.min(...validValues) : 0;
  const max = validValues.length ? Math.max(...validValues) : 1;
  const range = max - min || 1;

  const paths = useMemo(
    () =>
      series.map((item) => {
        const points = item.values.map((value, index) => {
          const x = left + (index / Math.max(item.values.length - 1, 1)) * (width - left - right);
          const y = top + (1 - (value - min) / range) * (height - top - bottom);
          return [x, y] as const;
        });
        const d = points.map((point, index) => (index ? 'L' : 'M') + point[0].toFixed(1) + ' ' + point[1].toFixed(1)).join(' ');
        return { ...item, d };
      }),
    [height, min, range, series],
  );

  return (
    <section className={cn('trend-chart', className)}>
      <header className="trend-chart-header">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <div className="trend-legend">
          {series.map((item) => <span key={item.id}><i style={{ background: item.color }} />{item.label}</span>)}
        </div>
      </header>
      <div className="trend-canvas">
        <svg aria-label={title} preserveAspectRatio="none" role="img" viewBox={'0 0 ' + width + ' ' + height}>
          {[0, 1, 2, 3].map((line) => {
            const y = top + line * ((height - top - bottom) / 3);
            return <line key={line} x1={left} x2={width - right} y1={y} y2={y} />;
          })}
          {paths.map((item) => (
            <motion.path
              animate={{ pathLength: 1, opacity: 1 }}
              d={item.d}
              fill="none"
              initial={{ pathLength: reduceMotion ? 1 : 0, opacity: reduceMotion ? 1 : 0.3 }}
              key={item.id}
              stroke={item.color}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              transition={{ duration: reduceMotion ? 0 : 0.55 }}
            />
          ))}
        </svg>
        <div className="trend-axis"><span>60s ago</span><span>now</span></div>
      </div>
    </section>
  );
}
