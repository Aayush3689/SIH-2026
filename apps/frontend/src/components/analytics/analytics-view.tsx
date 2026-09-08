'use client';

import { useId, useMemo, useState } from 'react';
import { useReducedMotion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  CalendarDays,
  Cpu,
  Plane,
  Radio,
  RotateCcw,
  TrendingUp,
} from 'lucide-react';
import { useApp } from '@/components/layout/app-providers';
import { Badge, Button, Card, CardHeader, Select } from '@/components/ui';
import { demoService } from '@/lib/services';
import { cn, formatNumber } from '@/lib/utils';
import type { TelemetrySample, TelemetrySeries } from '@/types/telemetry';
import type { Uav, UavStatus } from '@/types/uav';
import styles from './analytics-view.module.css';

type DateRange = '6h' | '24h' | '7d' | '30d';

interface RangeOption {
  id: DateRange;
  label: string;
  hours: number;
  points: number;
}

interface AnalyticsSignals {
  health: number;
  readiness: number;
  coverage: number;
  averageLoad: number;
  activeAlerts: number;
  monitoredAirframes: number;
  telemetryPoints: number;
}

interface ChartPoint {
  label: string;
  health: number;
  load: number;
  coverage: number;
  alerts: number;
}

type DistributionTone = 'blue' | 'cyan' | 'green' | 'amber' | 'red';

interface DistributionBar {
  label: string;
  detail: string;
  value: number;
  percentage: number;
  tone: DistributionTone;
}

interface OperationalDistribution {
  faults: DistributionBar[];
  rulBuckets: DistributionBar[];
  uptime: number;
  telemetryRate: number;
  missionUtilization: number;
}

const ALL_UAVS = '__all_uavs__';
const ALL_ENGINES = '__all_engines__';

const rangeOptions: RangeOption[] = [
  { id: '6h', label: 'Last 6 hours', hours: 6, points: 7 },
  { id: '24h', label: 'Last 24 hours', hours: 24, points: 9 },
  { id: '7d', label: 'Last 7 days', hours: 24 * 7, points: 8 },
  { id: '30d', label: 'Last 30 days', hours: 24 * 30, points: 10 },
];

const healthByStatus: Record<UavStatus, number> = {
  active: 90,
  warning: 72,
  critical: 46,
  offline: 88,
};

const readinessByStatus: Record<UavStatus, number> = {
  active: 95,
  warning: 71,
  critical: 32,
  offline: 84,
};

const coverageByStatus: Record<UavStatus, number> = {
  active: 99.4,
  warning: 97.2,
  critical: 94.8,
  offline: 98.1,
};

const clamp = (value: number, minimum: number, maximum: number) => Math.min(Math.max(value, minimum), maximum);

const average = (values: number[], fallback: number) =>
  values.length ? values.reduce((total, value) => total + value, 0) / values.length : fallback;

const latestSample = (series: TelemetrySeries | undefined): TelemetrySample | undefined =>
  series?.samples[series.samples.length - 1];

const hashScope = (value: string) => {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) % 10_007;
  }
  return hash;
};

function healthForUav(uav: Uav, sample?: TelemetrySample) {
  if (!sample) return healthByStatus[uav.status];

  const vibrationPenalty = Math.max(0, sample.engine_vibration_mm_s - 3) * 5.1;
  const thermalPenalty = Math.max(0, sample.cht_c - 175) * 0.25 + Math.max(0, sample.egt_c - 650) * 0.055;
  const pressurePenalty = Math.max(0, 350 - sample.oil_pressure_kpa) * 0.075;
  const statusAdjustment = uav.status === 'warning' ? -3 : uav.status === 'critical' ? -8 : 0;

  return clamp(Math.round(95 - vibrationPenalty - thermalPenalty - pressurePenalty + statusAdjustment), 20, 98);
}

function makeAxisLabel(timestamp: number, range: DateRange) {
  const date = new Date(timestamp);
  if (range === '6h' || range === '24h') {
    return new Intl.DateTimeFormat('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'Asia/Kolkata',
    }).format(date);
  }

  return new Intl.DateTimeFormat('en-IN', {
    day: '2-digit',
    month: 'short',
    timeZone: 'Asia/Kolkata',
  }).format(date);
}

function buildTrend(
  range: RangeOption,
  signals: AnalyticsSignals,
  generatedAt: string,
  scopeKey: string,
): ChartPoint[] {
  const anchor = Date.parse(generatedAt);
  const seed = hashScope(scopeKey);
  const direction = (seed % 9) - 4;

  return Array.from({ length: range.points }, (_, index) => {
    const progress = range.points === 1 ? 1 : index / (range.points - 1);
    const historicalBias = (1 - progress) * direction;
    const healthWave = Math.sin((index + (seed % 5)) * 0.95) * 1.45;
    const loadWave = Math.cos((index + (seed % 7)) * 0.82) * 5.6;
    const coverageWave = Math.sin((index + (seed % 3)) * 0.7) * 0.8;
    const alertPulse = Math.sin((index + seed) * 1.19) > 0.55 ? 1 : 0;
    const timestamp = anchor - range.hours * 60 * 60 * 1000 + progress * range.hours * 60 * 60 * 1000;

    return {
      label: makeAxisLabel(timestamp, range.id),
      health: Math.round(clamp(signals.health + historicalBias + healthWave, 20, 100)),
      load: Math.round(clamp(signals.averageLoad + loadWave + historicalBias * 0.9, 0, 100)),
      coverage: Number(clamp(signals.coverage + coverageWave + historicalBias * 0.15, 75, 100).toFixed(1)),
      alerts: Math.max(0, Math.round(signals.activeAlerts / Math.max(4, range.points / 2) + alertPulse)),
    };
  });
}

function buildOperationalDistribution(
  range: RangeOption,
  signals: AnalyticsSignals,
  scopeKey: string,
): OperationalDistribution {
  const scopeSize = signals.monitoredAirframes;
  const seed = hashScope(scopeKey);
  const periodBias = range.id === '30d' ? 6 : range.id === '7d' ? 3 : range.id === '24h' ? 1 : 0;
  const noScope = scopeSize === 0;

  const rawFaults = noScope
    ? [0, 0, 0]
    : [
        18 + (100 - signals.health) * 0.67 + (seed % 9) + periodBias,
        16 + (100 - signals.readiness) * 0.45 + ((seed >> 2) % 8) + periodBias * 0.5,
        13 + signals.activeAlerts * 5 + ((seed >> 4) % 7) + periodBias * 0.35,
      ];
  const faultTotal = rawFaults.reduce((total, value) => total + value, 0);
  const faultPercentage = (value: number) => (faultTotal ? Math.round((value / faultTotal) * 100) : 0);

  const highRisk = noScope
    ? 0
    : clamp(Math.round((76 - signals.health) / 20) + ((seed + periodBias) % 3 === 0 ? 1 : 0), 0, scopeSize);
  const watchRisk = noScope
    ? 0
    : clamp(
        Math.round((88 - signals.health) / 19) + (signals.activeAlerts > 0 ? 1 : 0),
        0,
        scopeSize - highRisk,
      );
  const stableRisk = Math.max(0, scopeSize - highRisk - watchRisk);
  const assetPercentage = (count: number) => (scopeSize ? Math.round((count / scopeSize) * 100) : 0);
  const assetDetail = (count: number) => `${count} asset${count === 1 ? '' : 's'}`;

  return {
    faults: [
      {
        label: 'Vibration',
        detail: 'Signature share',
        value: faultPercentage(rawFaults[0]),
        percentage: faultPercentage(rawFaults[0]),
        tone: 'red',
      },
      {
        label: 'Thermal',
        detail: 'Signature share',
        value: faultPercentage(rawFaults[1]),
        percentage: faultPercentage(rawFaults[1]),
        tone: 'amber',
      },
      {
        label: 'Lubrication',
        detail: 'Signature share',
        value: faultPercentage(rawFaults[2]),
        percentage: faultPercentage(rawFaults[2]),
        tone: 'blue',
      },
    ],
    rulBuckets: [
      {
        label: '≤ 25 h',
        detail: assetDetail(highRisk),
        value: highRisk,
        percentage: assetPercentage(highRisk),
        tone: 'red',
      },
      {
        label: '26–75 h',
        detail: assetDetail(watchRisk),
        value: watchRisk,
        percentage: assetPercentage(watchRisk),
        tone: 'amber',
      },
      {
        label: '> 75 h',
        detail: assetDetail(stableRisk),
        value: stableRisk,
        percentage: assetPercentage(stableRisk),
        tone: 'green',
      },
    ],
    uptime: noScope ? 0 : Number(clamp(signals.coverage + 0.6 + (seed % 11) / 10 - periodBias * 0.1, 80, 100).toFixed(1)),
    telemetryRate: noScope ? 0 : Math.max(0, Math.round(72 + signals.averageLoad * 0.86 + (seed % 17) - periodBias)),
    missionUtilization: noScope
      ? 0
      : Math.round(clamp(signals.readiness - 7 + signals.averageLoad * 0.19 + ((seed >> 1) % 7) - periodBias, 0, 100)),
  };
}

function scopeLabel(uavs: Uav[], selectedUavId: string, selectedEngineId: string) {
  if (selectedEngineId !== ALL_ENGINES) return selectedEngineId;
  if (selectedUavId !== ALL_UAVS) return uavs.find((uav) => uav.id === selectedUavId)?.callsign || selectedUavId;
  return 'Entire fleet';
}

function toneForHealth(value: number) {
  if (value >= 80) return 'green' as const;
  if (value >= 65) return 'amber' as const;
  return 'red' as const;
}

function toneForAlerts(value: number) {
  if (value === 0) return 'green' as const;
  if (value === 1) return 'amber' as const;
  return 'red' as const;
}

type TrendMetric = 'health' | 'load' | 'coverage';

interface SvgTrendChartProps {
  ariaLabel: string;
  color: string;
  data: ChartPoint[];
  domain: readonly [number, number];
  fill?: boolean;
  formatValue: (value: number) => string;
  metric: TrendMetric;
  reduceMotion: boolean | null;
}

interface SvgBarChartProps {
  ariaLabel: string;
  color: string;
  data: ChartPoint[];
  formatValue: (value: number) => string;
  reduceMotion: boolean | null;
}

interface SvgCoordinate {
  label: string;
  value: number;
  x: number;
  y: number;
}

const chartFrame = {
  width: 640,
  height: 230,
  left: 43,
  right: 14,
  top: 12,
  bottom: 31,
};

const chartPlotWidth = chartFrame.width - chartFrame.left - chartFrame.right;
const chartPlotHeight = chartFrame.height - chartFrame.top - chartFrame.bottom;
const chartBaseY = chartFrame.height - chartFrame.bottom;

function tickIndexes(length: number, maximumTicks = 5) {
  if (length <= maximumTicks) return Array.from({ length }, (_, index) => index);

  const indexes = new Set<number>([0, length - 1]);
  const interval = (length - 1) / (maximumTicks - 1);
  for (let index = 1; index < maximumTicks - 1; index += 1) {
    indexes.add(Math.round(interval * index));
  }
  return Array.from(indexes).sort((first, second) => first - second);
}

function horizontalTicks(domain: readonly [number, number], count = 4) {
  const [minimum, maximum] = domain;
  return Array.from({ length: count + 1 }, (_, index) => maximum - ((maximum - minimum) / count) * index);
}

function xPosition(index: number, length: number) {
  if (length <= 1) return chartFrame.left + chartPlotWidth / 2;
  return chartFrame.left + (index / (length - 1)) * chartPlotWidth;
}

function yPosition(value: number, domain: readonly [number, number]) {
  const [minimum, maximum] = domain;
  const range = Math.max(1, maximum - minimum);
  return chartFrame.top + ((maximum - clamp(value, minimum, maximum)) / range) * chartPlotHeight;
}

function xTextAnchor(index: number, length: number): 'start' | 'middle' | 'end' {
  if (index === 0) return 'start';
  if (index === length - 1) return 'end';
  return 'middle';
}

function SvgTrendChart({
  ariaLabel,
  color,
  data,
  domain,
  fill = false,
  formatValue,
  metric,
  reduceMotion,
}: SvgTrendChartProps) {
  const gradientId = `analytics-fill-${useId().replace(/:/g, '')}`;

  if (data.length === 0) {
    return <div className={styles.chartEmpty}>No chart data in the selected scope.</div>;
  }

  const coordinates: SvgCoordinate[] = data.map((point, index) => ({
    label: point.label,
    value: point[metric],
    x: xPosition(index, data.length),
    y: yPosition(point[metric], domain),
  }));
  const first = coordinates[0];
  const last = coordinates[coordinates.length - 1];
  const linePath = coordinates
    .map((point, index) => `${index === 0 ? 'M' : 'L'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ');
  const areaPath = `${linePath} L${last.x.toFixed(2)} ${chartBaseY} L${first.x.toFixed(2)} ${chartBaseY} Z`;
  const visibleIndexes = tickIndexes(data.length);

  return (
    <div className={cn(styles.localChart, reduceMotion && styles.localChartNoMotion)}>
      <svg
        aria-label={ariaLabel}
        className={styles.chartSvg}
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${chartFrame.width} ${chartFrame.height}`}
      >
        <title>{ariaLabel}</title>
        <desc>Values update with the selected date range, UAV, and engine filters.</desc>
        {fill ? (
          <defs>
            <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.25" />
              <stop offset="100%" stopColor={color} stopOpacity="0.015" />
            </linearGradient>
          </defs>
        ) : null}
        <g aria-hidden="true">
          {horizontalTicks(domain).map((value) => {
            const y = yPosition(value, domain);
            return (
              <g key={value}>
                <line className={styles.chartGridLine} x1={chartFrame.left} x2={chartFrame.width - chartFrame.right} y1={y} y2={y} />
                <text className={styles.chartAxisText} textAnchor="end" x={chartFrame.left - 8} y={y + 3}>{formatValue(value)}</text>
              </g>
            );
          })}
          <line className={styles.chartAxisLine} x1={chartFrame.left} x2={chartFrame.width - chartFrame.right} y1={chartBaseY} y2={chartBaseY} />
          {visibleIndexes.map((index) => {
            const point = coordinates[index];
            return (
              <text
                className={styles.chartAxisText}
                key={point.label}
                textAnchor={xTextAnchor(index, data.length)}
                x={point.x}
                y={chartFrame.height - 8}
              >
                {point.label}
              </text>
            );
          })}
        </g>
        {fill ? <path className={styles.chartArea} d={areaPath} fill={`url(#${gradientId})`} /> : null}
        <path className={styles.chartLine} d={linePath} pathLength={1} stroke={color} />
        {coordinates.map((point) => (
          <g className={styles.chartPointGroup} key={`${point.label}-${point.value}`}>
            <circle className={styles.chartPointTarget} cx={point.x} cy={point.y} r={8}>
              <title>{`${point.label}: ${formatValue(point.value)}`}</title>
            </circle>
            <circle className={styles.chartPointDot} cx={point.x} cy={point.y} fill={color} r={3.25} />
          </g>
        ))}
      </svg>
    </div>
  );
}

function SvgBarChart({ ariaLabel, color, data, formatValue, reduceMotion }: SvgBarChartProps) {
  const highestValue = Math.max(1, ...data.map((point) => point.alerts));
  const domain: readonly [number, number] = [0, Math.max(2, Math.ceil(highestValue * 1.25))];
  const slotWidth = data.length ? chartPlotWidth / data.length : chartPlotWidth;
  const barWidth = clamp(slotWidth * 0.58, 7, 27);
  const visibleIndexes = tickIndexes(data.length, 4);

  return (
    <div className={cn(styles.localChart, reduceMotion && styles.localChartNoMotion)}>
      <svg
        aria-label={ariaLabel}
        className={styles.chartSvg}
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${chartFrame.width} ${chartFrame.height}`}
      >
        <title>{ariaLabel}</title>
        <desc>Open alert events across the selected filter scope.</desc>
        <g aria-hidden="true">
          {horizontalTicks(domain, 3).map((value) => {
            const y = yPosition(value, domain);
            return (
              <g key={value}>
                <line className={styles.chartGridLine} x1={chartFrame.left} x2={chartFrame.width - chartFrame.right} y1={y} y2={y} />
                <text className={styles.chartAxisText} textAnchor="end" x={chartFrame.left - 8} y={y + 3}>{formatValue(value)}</text>
              </g>
            );
          })}
          <line className={styles.chartAxisLine} x1={chartFrame.left} x2={chartFrame.width - chartFrame.right} y1={chartBaseY} y2={chartBaseY} />
          {visibleIndexes.map((index) => {
            const point = data[index];
            return (
              <text
                className={styles.chartAxisText}
                key={point.label}
                textAnchor={xTextAnchor(index, data.length)}
                x={xPosition(index, data.length)}
                y={chartFrame.height - 8}
              >
                {point.label}
              </text>
            );
          })}
        </g>
        {data.map((point, index) => {
          const x = chartFrame.left + slotWidth * index + (slotWidth - barWidth) / 2;
          const y = yPosition(point.alerts, domain);
          const height = Math.max(0, chartBaseY - y);
          return (
            <rect
              className={styles.chartBar}
              fill={color}
              height={height}
              key={`${point.label}-${point.alerts}`}
              rx={2.5}
              ry={2.5}
              style={{ animationDelay: `${index * 30}ms` }}
              width={barWidth}
              x={x}
              y={y}
            >
              <title>{`${point.label}: ${formatValue(point.alerts)}`}</title>
            </rect>
          );
        })}
      </svg>
    </div>
  );
}

export function AnalyticsView() {
  const { alerts, uavs } = useApp();
  const reduceMotion = useReducedMotion();
  const [dateRange, setDateRange] = useState<DateRange>('24h');
  const [selectedUavId, setSelectedUavId] = useState(ALL_UAVS);
  const [selectedEngineId, setSelectedEngineId] = useState(ALL_ENGINES);
  const fleetAnalytics = useMemo(() => demoService.getAnalytics(), []);
  const telemetrySeries = useMemo(() => demoService.listTelemetry(), []);

  const availableEngines = useMemo(
    () =>
      uavs.filter((uav) => selectedUavId === ALL_UAVS || uav.id === selectedUavId),
    [selectedUavId, uavs],
  );

  const scopedUavs = useMemo(
    () =>
      uavs.filter(
        (uav) =>
          (selectedUavId === ALL_UAVS || uav.id === selectedUavId) &&
          (selectedEngineId === ALL_ENGINES || uav.engine_id === selectedEngineId),
      ),
    [selectedEngineId, selectedUavId, uavs],
  );

  const scopedTelemetry = useMemo(
    () =>
      telemetrySeries.filter(
        (series) =>
          scopedUavs.some((uav) => uav.id === series.uav_id) &&
          (selectedEngineId === ALL_ENGINES || latestSample(series)?.engine_id === selectedEngineId),
      ),
    [scopedUavs, selectedEngineId, telemetrySeries],
  );

  const signals = useMemo<AnalyticsSignals>(() => {
    const healthScores = scopedUavs.map((uav) => {
      const telemetry = scopedTelemetry.find((series) => series.uav_id === uav.id);
      return healthForUav(uav, latestSample(telemetry));
    });
    const latestSamples = scopedTelemetry
      .map((series) => latestSample(series))
      .filter((sample): sample is TelemetrySample => Boolean(sample));
    const matchingAlerts = alerts.filter(
      (alert) =>
        alert.status !== 'resolved' &&
        scopedUavs.some(
          (uav) =>
            alert.uav_id === uav.id && (selectedEngineId === ALL_ENGINES || alert.engine_id === selectedEngineId),
        ),
    );

    return {
      health: Math.round(average(healthScores, fleetAnalytics.fleet_health_score)),
      readiness: Math.round(
        average(scopedUavs.map((uav) => readinessByStatus[uav.status]), fleetAnalytics.mission_readiness_pct),
      ),
      coverage: Number(
        average(scopedUavs.map((uav) => coverageByStatus[uav.status]), fleetAnalytics.telemetry_ingestion_pct).toFixed(1),
      ),
      averageLoad: Math.round(average(latestSamples.map((sample) => sample.engine_load_pct), 0)),
      activeAlerts: matchingAlerts.length,
      monitoredAirframes: scopedUavs.length,
      telemetryPoints: scopedTelemetry.reduce((total, series) => total + series.samples.length, 0),
    };
  }, [alerts, fleetAnalytics, scopedTelemetry, scopedUavs, selectedEngineId]);

  const activeRange = rangeOptions.find((option) => option.id === dateRange) || rangeOptions[1];
  const activeScopeLabel = scopeLabel(uavs, selectedUavId, selectedEngineId);
  const filterKey = [dateRange, selectedUavId, selectedEngineId].join(':');
  const chartData = useMemo(
    () => buildTrend(activeRange, signals, fleetAnalytics.generated_at, filterKey),
    [activeRange, filterKey, fleetAnalytics.generated_at, signals],
  );
  const operationalDistribution = useMemo(
    () => buildOperationalDistribution(activeRange, signals, filterKey),
    [activeRange, filterKey, signals],
  );

  const summaryCards = [
    {
      label: 'Health score',
      value: `${signals.health}%`,
      detail: signals.monitoredAirframes === 1 ? '1 monitored airframe' : `${signals.monitoredAirframes} monitored airframes`,
      Icon: Activity,
      tone: toneForHealth(signals.health),
    },
    {
      label: 'Mission readiness',
      value: `${signals.readiness}%`,
      detail: 'Readiness across selected assets',
      Icon: Plane,
      tone: toneForHealth(signals.readiness),
    },
    {
      label: 'Open alerts',
      value: String(signals.activeAlerts),
      detail: signals.activeAlerts === 1 ? '1 active condition' : `${signals.activeAlerts} active conditions`,
      Icon: AlertTriangle,
      tone: toneForAlerts(signals.activeAlerts),
    },
    {
      label: 'Telemetry capture',
      value: `${formatNumber(signals.coverage, 1)}%`,
      detail: `${formatNumber(signals.telemetryPoints)} samples in scope`,
      Icon: Radio,
      tone: 'cyan' as const,
    },
  ];

  const handleUavChange = (value: string) => {
    setSelectedUavId(value);
    if (value !== ALL_UAVS) {
      const selected = uavs.find((uav) => uav.id === value);
      if (selected && selectedEngineId !== ALL_ENGINES && selected.engine_id !== selectedEngineId) {
        setSelectedEngineId(ALL_ENGINES);
      }
    }
  };

  const handleEngineChange = (value: string) => {
    setSelectedEngineId(value);
    if (value !== ALL_ENGINES) {
      const matchingUav = uavs.find((uav) => uav.engine_id === value);
      if (matchingUav) setSelectedUavId(matchingUav.id);
    }
  };

  const resetFilters = () => {
    setDateRange('24h');
    setSelectedUavId(ALL_UAVS);
    setSelectedEngineId(ALL_ENGINES);
  };

  return (
    <section aria-labelledby="analytics-title" className={cn('page-enter', styles.analytics)}>
      <header className={styles.heading}>
        <div>
          <div className="eyebrow">Operational intelligence</div>
          <h1 id="analytics-title">Analytics</h1>
          <p>Review fleet reliability, utilization, and telemetry quality across a selected analysis window.</p>
        </div>
        <Badge dot tone="cyan">Demo snapshot</Badge>
      </header>

      <Card className={styles.filters}>
        <div className={styles.filterTitle}>
          <span className={styles.filterIcon}><CalendarDays size={16} /></span>
          <div>
            <strong>Analysis filters</strong>
            <span>Changes update the trend views below.</span>
          </div>
        </div>
        <div className={styles.filterControls}>
          <Select
            aria-label="Date range"
            className={styles.filterField}
            label="Date range"
            onChange={(event) => setDateRange(event.target.value as DateRange)}
            value={dateRange}
          >
            {rangeOptions.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
          </Select>
          <Select
            aria-label="UAV"
            className={styles.filterField}
            label="UAV"
            onChange={(event) => handleUavChange(event.target.value)}
            value={selectedUavId}
          >
            <option value={ALL_UAVS}>All UAVs</option>
            {uavs.map((uav) => <option key={uav.id} value={uav.id}>{uav.callsign} · {uav.id}</option>)}
          </Select>
          <Select
            aria-label="Engine"
            className={styles.filterField}
            label="Engine"
            onChange={(event) => handleEngineChange(event.target.value)}
            value={selectedEngineId}
          >
            <option value={ALL_ENGINES}>All engines</option>
            {availableEngines.map((uav) => <option key={uav.engine_id} value={uav.engine_id}>{uav.engine_id}</option>)}
          </Select>
          <Button className={styles.resetButton} onClick={resetFilters} size="sm" variant="ghost">
            <RotateCcw size={14} /> Reset
          </Button>
        </div>
      </Card>

      <div aria-live="polite" className={styles.scopeLine}>
        <TrendingUp size={15} />
        <span>Viewing <b>{activeScopeLabel}</b> · {activeRange.label.toLowerCase()}</span>
      </div>

      <div className={styles.summaryGrid}>
        {summaryCards.map(({ detail, Icon, label, tone, value }) => (
          <Card className={styles.summaryCard} key={label}>
            <div className={styles.summaryTop}>
              <span>{label}</span>
              <span className={cn(styles.summaryIcon, styles[`summaryIcon${tone.charAt(0).toUpperCase()}${tone.slice(1)}`])}>
                <Icon size={16} />
              </span>
            </div>
            <strong>{value}</strong>
            <small>{detail}</small>
          </Card>
        ))}
      </div>

      <div className={styles.primaryGrid}>
        <Card className={styles.chartCard}>
          <CardHeader
            subtitle="Normalized operational health across the selected scope"
            title="Health score trend"
            action={<Badge tone={toneForHealth(signals.health)}>{signals.health >= 80 ? 'Stable' : signals.health >= 65 ? 'Watch' : 'Action needed'}</Badge>}
          />
          <div className={styles.chartWrap}>
            <SvgTrendChart
              ariaLabel="Health score trend chart"
              color="#59a1ff"
              data={chartData}
              domain={[0, 100]}
              formatValue={(value) => `${value}%`}
              key={`health-${filterKey}`}
              metric="health"
              reduceMotion={reduceMotion}
            />
          </div>
          <footer className={styles.chartFooter}>
            <span><i className={styles.blueDot} />Current operational health</span>
            <span>Updated {new Intl.DateTimeFormat('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' }).format(new Date(fleetAnalytics.generated_at))} IST</span>
          </footer>
        </Card>

        <Card className={styles.alertCard}>
          <CardHeader
            subtitle="Unresolved conditions per analysis interval"
            title="Alert exposure"
            action={<Badge tone={toneForAlerts(signals.activeAlerts)}>{signals.activeAlerts} open</Badge>}
          />
          <div className={styles.chartWrap}>
            <SvgBarChart
              ariaLabel="Alert exposure bar chart"
              color="#f1b955"
              data={chartData}
              formatValue={(value) => `${value} open conditions`}
              key={`alerts-${filterKey}`}
              reduceMotion={reduceMotion}
            />
          </div>
          <footer className={styles.alertFooter}>
            <AlertTriangle size={14} />
            <span>{signals.activeAlerts === 0 ? 'No active conditions in this scope.' : 'Escalate unresolved conditions through Alerts.'}</span>
          </footer>
        </Card>
      </div>

      <div className={styles.secondaryGrid}>
        <Card className={styles.chartCard}>
          <CardHeader
            subtitle="Mean engine load for available telemetry samples"
            title="Operating load"
            action={<span className={styles.chartValue}>{signals.averageLoad}% mean</span>}
          />
          <div className={styles.chartWrap}>
            <SvgTrendChart
              ariaLabel="Operating load area chart"
              color="#39d0c8"
              data={chartData}
              domain={[0, 100]}
              fill
              formatValue={(value) => `${value}%`}
              key={`load-${filterKey}`}
              metric="load"
              reduceMotion={reduceMotion}
            />
          </div>
        </Card>

        <Card className={styles.chartCard}>
          <CardHeader
            subtitle="Percentage of expected telemetry retained"
            title="Telemetry coverage"
            action={<Badge tone="cyan">{formatNumber(signals.coverage, 1)}%</Badge>}
          />
          <div className={styles.chartWrap}>
            <SvgTrendChart
              ariaLabel="Telemetry coverage trend chart"
              color="#39d0c8"
              data={chartData}
              domain={[75, 100]}
              formatValue={(value) => `${value}%`}
              key={`coverage-${filterKey}`}
              metric="coverage"
              reduceMotion={reduceMotion}
            />
          </div>
        </Card>
      </div>

      <Card className={styles.distributionCard}>
        <CardHeader
          subtitle="Derived local signals across the active analysis scope"
          title="Operational distribution"
          action={<Badge tone="blue">{activeRange.label}</Badge>}
        />
        <div className={styles.distributionGrid}>
          <section aria-label="Fault distribution" className={styles.distributionSection}>
            <header className={styles.distributionHeading}>
              <span>Fault distribution</span>
              <small>Signal share</small>
            </header>
            <div className={styles.distributionRows}>
              {operationalDistribution.faults.map((item) => (
                <div className={styles.distributionRow} key={item.label}>
                  <div className={styles.distributionRowTop}>
                    <span>{item.label}</span>
                    <b>{item.value}%</b>
                  </div>
                  <div className={styles.distributionTrack}>
                    <span
                      className={cn(styles.distributionFill, styles[`distributionFill${item.tone.charAt(0).toUpperCase()}${item.tone.slice(1)}`])}
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                  <small>{item.detail}</small>
                </div>
              ))}
            </div>
          </section>

          <section aria-label="Remaining useful life risk buckets" className={styles.distributionSection}>
            <header className={styles.distributionHeading}>
              <span>RUL risk buckets</span>
              <small>Projected assets</small>
            </header>
            <div className={styles.distributionRows}>
              {operationalDistribution.rulBuckets.map((item) => (
                <div className={styles.distributionRow} key={item.label}>
                  <div className={styles.distributionRowTop}>
                    <span>{item.label}</span>
                    <b>{item.value}</b>
                  </div>
                  <div className={styles.distributionTrack}>
                    <span
                      className={cn(styles.distributionFill, styles[`distributionFill${item.tone.charAt(0).toUpperCase()}${item.tone.slice(1)}`])}
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                  <small>{item.detail}</small>
                </div>
              ))}
            </div>
          </section>

          <section aria-label="Connection and utilization indicators" className={styles.distributionSection}>
            <header className={styles.distributionHeading}>
              <span>Connection &amp; utilization</span>
              <small>Selected scope</small>
            </header>
            <div className={styles.indicatorList}>
              <div className={styles.indicatorRow}>
                <span><i className={cn(styles.indicatorDot, styles.indicatorDotGreen)} />Connection uptime</span>
                <b>{formatNumber(operationalDistribution.uptime, 1)}%</b>
              </div>
              <div className={styles.indicatorRow}>
                <span><i className={cn(styles.indicatorDot, styles.indicatorDotCyan)} />Telemetry rate</span>
                <b>{formatNumber(operationalDistribution.telemetryRate)} frames/s</b>
              </div>
              <div className={styles.indicatorRow}>
                <span><i className={cn(styles.indicatorDot, styles.indicatorDotBlue)} />Mission utilization</span>
                <b>{operationalDistribution.missionUtilization}%</b>
              </div>
            </div>
          </section>
        </div>
      </Card>

      <Card className={styles.insightCard}>
        <div className={styles.insightIcon}><Cpu size={18} /></div>
        <div>
          <span className={styles.insightLabel}>Scope note</span>
          <strong>{signals.monitoredAirframes === 0 ? 'No airframes match the current filter combination.' : `${signals.monitoredAirframes} airframe${signals.monitoredAirframes === 1 ? '' : 's'} and ${signals.telemetryPoints} telemetry samples are represented in this view.`}</strong>
        </div>
        <span className={styles.insightDetail}>Values are generated from the local demo snapshot and update with the selected filters.</span>
      </Card>
    </section>
  );
}

export default AnalyticsView;
