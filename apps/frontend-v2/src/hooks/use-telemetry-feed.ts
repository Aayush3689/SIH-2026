'use client';

import { useEffect, useMemo, useState } from 'react';
import { demoService } from '@/lib/services';
import type { TelemetrySample } from '@/types/telemetry';

function updateTimestamp(sample: TelemetrySample, offset: number, engineId?: string): TelemetrySample {
  return {
    ...sample,
    engine_id: engineId || sample.engine_id,
    elapsed_s: sample.elapsed_s + offset * 5,
    timestamp: new Date().toISOString(),
  };
}

export function useTelemetryFeed(uavId: string, engineId?: string, interval = 1400) {
  const base = useMemo(() => {
    const direct = demoService.getTelemetry(uavId);
    return direct.length ? direct : demoService.getTelemetry('UAV-001');
  }, [uavId]);
  const [history, setHistory] = useState<TelemetrySample[]>(() => base.slice(-30));
  const [, setCursor] = useState(0);
  const [samplesReceived, setSamplesReceived] = useState(1);

  useEffect(() => {
    setHistory(base.slice(-30));
    setCursor(0);
    setSamplesReceived(1);
  }, [base, uavId]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setCursor((current) => {
        const nextCursor = current + 1;
        const source = base[nextCursor % base.length];
        if (source) {
          setHistory((currentHistory) => [
            ...currentHistory.slice(-59),
            updateTimestamp(source, nextCursor, engineId),
          ]);
        }
        return nextCursor;
      });
      setSamplesReceived((current) => Math.min(current + 1, Math.max(base.length, 8)));
    }, interval);
    return () => window.clearInterval(timer);
  }, [base, engineId, interval]);

  const latest = history[history.length - 1];
  const inferenceInput = useMemo(
    () => history.slice(-Math.min(samplesReceived, history.length)),
    [history, samplesReceived],
  );
  const inference = useMemo(
    () => (inferenceInput.length ? demoService.getInference(inferenceInput) : null),
    [inferenceInput],
  );

  return {
    history,
    latest,
    inference,
    samplesReceived,
    isRulReady: samplesReceived >= 8,
  };
}
