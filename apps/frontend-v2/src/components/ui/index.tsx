'use client';

import {
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  useEffect,
} from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { LoaderCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  children,
  className,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn('button', 'button--' + variant, 'button--' + size, className)}
      disabled={disabled || loading}
      type={type}
      {...props}
    >
      {loading ? <LoaderCircle aria-hidden="true" className="spin" size={16} /> : null}
      <span>{children}</span>
    </button>
  );
}

export function IconButton({
  children,
  label,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button aria-label={label} className={cn('icon-button', className)} type="button" {...props}>
      {children}
    </button>
  );
}

export function Card({
  children,
  className,
  interactive = false,
}: {
  children: ReactNode;
  className?: string;
  interactive?: boolean;
}) {
  return <section className={cn('card', interactive && 'card--interactive', className)}>{children}</section>;
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="card-header">
      <div>
        <h2 className="card-title">{title}</h2>
        {subtitle ? <p className="card-subtitle">{subtitle}</p> : null}
      </div>
      {action ? <div className="card-action">{action}</div> : null}
    </div>
  );
}

type Tone = 'blue' | 'green' | 'amber' | 'red' | 'cyan' | 'gray' | 'purple';

export function Badge({
  children,
  tone = 'gray',
  dot = false,
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span className={cn('badge', 'badge--' + tone, className)}>
      {dot ? <span className="badge-dot" /> : null}
      {children}
    </span>
  );
}

export function Input({
  label,
  hint,
  error,
  className,
  id,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  label?: string;
  hint?: string;
  error?: string;
}) {
  const inputId = id || props.name;
  return (
    <label className={cn('field', className)} htmlFor={inputId}>
      {label ? <span className="field-label">{label}</span> : null}
      <input id={inputId} className={cn('input', error && 'input--error')} {...props} />
      {error ? <span className="field-error">{error}</span> : hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

export function Select({
  label,
  hint,
  error,
  children,
  className,
  id,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & {
  label?: string;
  hint?: string;
  error?: string;
  children: ReactNode;
}) {
  const selectId = id || props.name;
  return (
    <label className={cn('field', className)} htmlFor={selectId}>
      {label ? <span className="field-label">{label}</span> : null}
      <span className="select-wrap">
        <select id={selectId} className={cn('input', 'select', error && 'input--error')} {...props}>
          {children}
        </select>
      </span>
      {error ? <span className="field-error">{error}</span> : hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled = false,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  return (
    <label className={cn('toggle-row', disabled && 'is-disabled')}>
      <span>
        <span className="toggle-label">{label}</span>
        {description ? <span className="toggle-description">{description}</span> : null}
      </span>
      <input
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        type="checkbox"
      />
      <span aria-hidden="true" className="toggle-control" />
    </label>
  );
}

export function ProgressBar({
  value,
  tone = 'blue',
  label,
  showValue = true,
}: {
  value: number;
  tone?: Tone;
  label?: ReactNode;
  showValue?: boolean;
}) {
  const bounded = Math.max(0, Math.min(100, value));
  return (
    <div className="progress">
      {label || showValue ? (
        <div className="progress-meta">
          <span>{label}</span>
          {showValue ? <span>{bounded.toFixed(0)}%</span> : null}
        </div>
      ) : null}
      <div className="progress-track">
        <motion.span
          animate={{ width: bounded + '%' }}
          className={cn('progress-fill', 'progress-fill--' + tone)}
          initial={{ width: 0 }}
          transition={{ duration: 0.35 }}
        />
      </div>
    </div>
  );
}

export function Tabs({
  tabs,
  active,
  onChange,
  className,
}: {
  tabs: Array<{ id: string; label: ReactNode; count?: number }>;
  active: string;
  onChange: (id: string) => void;
  className?: string;
}) {
  return (
    <div aria-label="View selector" className={cn('tabs', className)} role="tablist">
      {tabs.map((tab) => (
        <button
          aria-selected={active === tab.id}
          className={cn('tab', active === tab.id && 'tab--active')}
          key={tab.id}
          onClick={() => onChange(tab.id)}
          role="tab"
          type="button"
        >
          {tab.label}
          {typeof tab.count === 'number' ? <span className="tab-count">{tab.count}</span> : null}
        </button>
      ))}
    </div>
  );
}

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  size = 'lg',
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
}) {
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose, open]);

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          animate={{ opacity: 1 }}
          className="modal-backdrop"
          exit={{ opacity: 0 }}
          initial={{ opacity: 0 }}
          onMouseDown={onClose}
        >
          <motion.section
            animate={{ opacity: 1, scale: 1, y: 0 }}
            aria-modal="true"
            className={cn('modal', 'modal--' + size)}
            exit={{ opacity: 0, scale: reduceMotion ? 1 : 0.98, y: reduceMotion ? 0 : 8 }}
            initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.98, y: reduceMotion ? 0 : 8 }}
            onMouseDown={(event) => event.stopPropagation()}
            role="dialog"
            transition={{ duration: reduceMotion ? 0 : 0.2 }}
          >
            <header className="modal-header">
              <div>
                <h2>{title}</h2>
                {description ? <p>{description}</p> : null}
              </div>
              <IconButton label="Close dialog" onClick={onClose}>
                <X size={19} />
              </IconButton>
            </header>
            <div className="modal-body">{children}</div>
          </motion.section>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {action ? <div className="section-heading-action">{action}</div> : null}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: ReactNode;
  description: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon ? <div className="empty-state-icon">{icon}</div> : null}
      <h3>{title}</h3>
      <p>{description}</p>
      {action ? <div>{action}</div> : null}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cn('skeleton', className)} />;
}
