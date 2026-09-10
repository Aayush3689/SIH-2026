'use client';

import { useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { CheckCircle2, CircleAlert, Database, Link2, RefreshCw, Save, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import { toast } from 'sonner';
import { demoService } from '@/lib/services';
import { Badge, Button, Card, CardHeader, SectionHeading } from '@/components/ui';
import { cn } from '@/lib/utils';

export function DataMappingView() {
  const reduceMotion = useReducedMotion();
  const incoming = useMemo(() => demoService.listIncomingTelemetryFields(), []);
  const targets = useMemo(() => demoService.listPlatformTelemetryParameters(), []);
  const [mapping, setMapping] = useState<Record<string, string>>(() =>
    Object.fromEntries(demoService.listMappings().map((item) => [item.target_field, item.source_field])),
  );
  const [validated, setValidated] = useState(false);

  const evaluation = useMemo(() => {
    const requiredTargets = targets.filter((target) => target.required);
    const mapped = targets.filter((target) => Boolean(mapping[target.target_field]));
    const invalid = targets.filter((target) => {
      const source = incoming.find((item) => item.source_field === mapping[target.target_field]);
      return Boolean(source && source.unit !== target.expected_unit);
    });
    const missingRequired = requiredTargets.filter((target) => !mapping[target.target_field]);
    return {
      required: requiredTargets.length,
      mapped: mapped.length,
      unmapped: targets.length - mapped.length,
      missingRequired: missingRequired.length,
      invalid: invalid.length,
      isValid: missingRequired.length === 0 && invalid.length === 0,
    };
  }, [incoming, mapping, targets]);

  const mapTarget = (targetField: string, sourceField: string) => {
    setValidated(false);
    setMapping((current) => ({ ...current, [targetField]: sourceField }));
  };

  const validate = () => {
    if (!evaluation.isValid) {
      toast.error('Mapping validation failed. Review highlighted parameters.');
      return;
    }
    setValidated(true);
    toast.success('Mapping validation passed.', { description: 'All required parameters match the platform schema.' });
  };

  const reset = () => {
    setMapping(Object.fromEntries(demoService.listMappings().map((item) => [item.target_field, item.source_field])));
    setValidated(false);
    toast.info('Demo mapping restored.');
  };

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="page-enter mapping-page"
      initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }}
      transition={{ duration: reduceMotion ? 0 : 0.2 }}
    >
      <SectionHeading
        action={
          <div className="heading-action-group">
            <Button onClick={reset} size="sm" variant="ghost"><RefreshCw size={14} /> Reset demo</Button>
            <Button disabled={!evaluation.isValid} onClick={validate} size="sm">
              <ShieldCheck size={15} /> Validate mapping
            </Button>
          </div>
        }
        description="Normalize raw adapter fields into the telemetry schema consumed by the operations console."
        eyebrow={<><Badge dot tone="cyan">Live schema preview</Badge><span className="heading-refresh">UAV-001 · GCS source</span></>}
        title="Data Mapping"
      />

      <div className="mapping-page-summary">
        <div><span>Required parameters</span><b>{evaluation.required}</b><small>Schema-critical fields</small></div>
        <div><span>Mapped parameters</span><b className="value-good">{evaluation.mapped}</b><small>Active transformations</small></div>
        <div><span>Unmapped parameters</span><b className={evaluation.unmapped ? 'value-amber' : 'value-good'}>{evaluation.unmapped}</b><small>Optional or remaining fields</small></div>
        <div><span>Invalid parameters</span><b className={evaluation.invalid ? 'value-bad' : 'value-good'}>{evaluation.invalid}</b><small>Unit/schema mismatch</small></div>
      </div>

      <Card className="mapping-page-card">
        <CardHeader
          action={<Badge dot tone={evaluation.isValid ? 'green' : 'amber'}>{evaluation.isValid ? 'Ready to validate' : 'Mapping required'}</Badge>}
          subtitle="Mapping changes are local to this frontend demo."
          title="Raw telemetry → platform parameters"
        />
        <div className="mapping-page-toolbar">
          <div className="mapping-toolbar-source"><Database size={15} /><span><b>GCS MAVLink adapter</b><small>Receiving 128 packets/s</small></span></div>
          <div className="mapping-toolbar-message"><Link2 size={14} /> Lines indicate active field transforms</div>
        </div>
        <div className="mapping-page-workspace">
          <div className="mapping-page-raw">
            <div className="mapping-page-heading"><span>Incoming / raw data</span><span>State</span></div>
            {incoming.map((field) => (
              <div className="mapping-page-raw-row" key={field.id}>
                <span className="mapping-page-field">
                  <b>{field.source_field}</b>
                  <small>{field.display_name} · {field.source} · {field.unit}</small>
                </span>
                <span className="mapping-page-live-value">{field.current_value} <i>{field.unit}</i></span>
                <Badge tone={field.status === 'valid' ? 'green' : field.status === 'invalid' ? 'red' : 'amber'}>{field.status}</Badge>
              </div>
            ))}
          </div>
          <div aria-hidden="true" className="mapping-page-lines">
            {targets.map((target, index) => mapping[target.target_field] ? (
              <motion.span
                animate={{ opacity: 1, scaleX: 1 }}
                className="mapping-page-line"
                initial={{ opacity: 0, scaleX: 0 }}
                key={target.target_field}
                style={{ top: 48 + index * 55 }}
                transition={{ duration: reduceMotion ? 0 : 0.28 }}
              />
            ) : null)}
          </div>
          <div className="mapping-page-target">
            <div className="mapping-page-heading"><span>Standardized parameter</span><span>Incoming field</span></div>
            {targets.map((target) => {
              const source = incoming.find((field) => field.source_field === mapping[target.target_field]);
              const mismatch = Boolean(source && source.unit !== target.expected_unit);
              return (
                <div className={cn('mapping-page-target-row', mismatch && 'mapping-page-target-row--invalid')} key={target.target_field}>
                  <span className="mapping-page-field">
                    <b>{target.display_name}</b>
                    <small>{target.target_field} · expected {target.expected_unit}</small>
                  </span>
                  <span className="mapping-required">{target.required ? 'Required' : 'Optional'}</span>
                  <select
                    aria-label={'Map source to ' + target.display_name}
                    className={cn('input', 'mapping-page-select', mismatch && 'input--error')}
                    onChange={(event) => mapTarget(target.target_field, event.target.value)}
                    value={mapping[target.target_field] || ''}
                  >
                    <option value="">Unmapped</option>
                    {incoming.map((field) => <option key={field.id} value={field.source_field}>{field.source_field} · {field.unit}</option>)}
                  </select>
                </div>
              );
            })}
          </div>
        </div>
        <div className="mapping-validation-panel">
          <div className="mapping-validation-copy">
            {validated ? <CheckCircle2 size={20} /> : evaluation.isValid ? <ShieldCheck size={20} /> : <CircleAlert size={20} />}
            <div>
              <strong>{validated ? 'Mapping validation passed' : evaluation.isValid ? 'Required field mapping complete' : 'Complete required mappings'}</strong>
              <span>{validated ? 'The schema is ready for the demo data source.' : evaluation.isValid ? 'Validate before saving a data source.' : 'Map all required parameters before validation.'}</span>
            </div>
          </div>
          <div className="mapping-validation-counts">
            <span><b>{evaluation.required}</b> required</span>
            <span><b>{evaluation.mapped}</b> mapped</span>
            <span><b className={evaluation.missingRequired ? 'value-bad' : 'value-good'}>{evaluation.missingRequired}</b> missing</span>
            <span><b className={evaluation.invalid ? 'value-bad' : 'value-good'}>{evaluation.invalid}</b> invalid</span>
          </div>
          <Button disabled={!evaluation.isValid} onClick={validate} variant={validated ? 'success' : 'primary'}>
            {validated ? <><CheckCircle2 size={15} /> Validated</> : <><Save size={15} /> Validate mapping</>}
          </Button>
        </div>
      </Card>

      <div className="mapping-guidance-grid">
        <Card className="mapping-guidance-card">
          <div className="guidance-icon"><SlidersHorizontal size={17} /></div>
          <div><strong>Unit checks</strong><p>Mappings are validated against expected platform units before the source can be saved.</p></div>
        </Card>
        <Card className="mapping-guidance-card">
          <div className="guidance-icon"><Link2 size={17} /></div>
          <div><strong>Adapter isolation</strong><p>The browser displays mock packets only; parsing and transport remain backend-adapter responsibilities.</p></div>
        </Card>
        <Card className="mapping-guidance-card">
          <div className="guidance-icon"><CheckCircle2 size={17} /></div>
          <div><strong>Schema readiness</strong><p>Validated mappings keep monitoring and AI inference fields consistent across source types.</p></div>
        </Card>
      </div>
    </motion.div>
  );
}
