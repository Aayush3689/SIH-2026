'use client';

import { type ReactNode, useEffect, useMemo, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import {
  Activity,
  AirVent,
  BarChart3,
  Bell,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Database,
  Gauge,
  Layers3,
  LayoutDashboard,
  Menu,
  Plane,
  Search,
  Settings,
  X,
} from 'lucide-react';
import { toast } from 'sonner';
import { AddUavWizard } from '@/components/uav/add-uav-wizard';
import { Badge, IconButton } from '@/components/ui';
import { useApp } from '@/components/layout/app-providers';
import { cn } from '@/lib/utils';

const navigation = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, detail: 'Fleet overview and operational summary' },
  { href: '/uav-fleet', label: 'UAV Fleet', icon: Plane, detail: 'Aircraft inventory and data sources' },
  { href: '/gateway', label: 'Gateway', icon: Database, detail: 'Adapters and connections' },
  { href: '/data-mapping', label: 'Data Mapping', icon: Layers3, detail: 'Normalize incoming telemetry' },
  { href: '/live-monitoring', label: 'Live Monitoring', icon: Activity, detail: 'Real-time engine telemetry' },
  { href: '/engine-health', label: 'Engine Health', icon: Gauge, detail: 'Predictions and remaining life' },
  { href: '/alerts', label: 'Alerts', icon: CircleAlert, detail: 'Operator alert workflow' },
  { href: '/analytics', label: 'Analytics', icon: BarChart3, detail: 'Fleet trends and insight' },
  { href: '/settings', label: 'Settings', icon: Settings, detail: 'Preferences and demo controls' },
];

function alertTone(severity: 'info' | 'warning' | 'critical') {
  if (severity === 'critical') return 'red' as const;
  if (severity === 'warning') return 'amber' as const;
  return 'blue' as const;
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { alerts, environment, selectedUavId, setSelectedUavId, uavs } = useApp();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    const updateLayout = () => {
      if (window.innerWidth < 1100) setCollapsed(true);
      if (window.innerWidth > 720) setMobileOpen(false);
    };
    updateLayout();
    window.addEventListener('resize', updateLayout);
    return () => window.removeEventListener('resize', updateLayout);
  }, []);

  const activeNav = navigation.find((item) => pathname === item.href || pathname.startsWith(item.href + '/')) || navigation[0];
  const openAlerts = alerts.filter((alert) => alert.status === 'open');
  const results = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return [];
    const navResults = navigation
      .filter((item) => (item.label + ' ' + item.detail).toLowerCase().includes(term))
      .map((item) => ({ kind: 'Navigate', title: item.label, subtitle: item.detail, href: item.href }));
    const uavResults = uavs
      .filter((uav) => (uav.id + ' ' + uav.callsign + ' ' + uav.engine_id).toLowerCase().includes(term))
      .map((uav) => ({
        kind: 'UAV',
        title: uav.id + ' · ' + uav.callsign,
        subtitle: uav.engine_id + ' · ' + uav.mission_status,
        href: '/uav-fleet',
        uavId: uav.id,
      }));
    const alertResults = alerts
      .filter((alert) => (alert.id + ' ' + alert.uav_id + ' ' + (alert.engine_id || '') + ' ' + alert.title).toLowerCase().includes(term))
      .map((alert) => ({
        kind: 'Alert',
        title: alert.id + ' · ' + alert.title,
        subtitle: alert.uav_id + ' · ' + alert.status,
        href: '/alerts',
        uavId: alert.uav_id,
      }));
    return [...navResults, ...uavResults, ...alertResults].slice(0, 7);
  }, [alerts, query, uavs]);

  const goToResult = (result: { href: string; uavId?: string }) => {
    if (result.uavId) setSelectedUavId(result.uavId);
    setQuery('');
    router.push(result.href);
  };

  const goTo = (href: string) => {
    setMobileOpen(false);
    router.push(href);
  };

  return (
    <div className={cn('app-shell', collapsed && 'app-shell--collapsed')}>
      {mobileOpen ? <button aria-label="Close navigation" className="mobile-nav-backdrop" onClick={() => setMobileOpen(false)} type="button" /> : null}
      <aside className={cn('sidebar', collapsed && 'sidebar--collapsed', mobileOpen && 'sidebar--mobile-open')}>
        <div className="brand">
          <div className="brand-mark"><AirVent size={22} /></div>
          <div className="brand-copy">
            <strong>AeroSentinel</strong>
            <span>Engine Operations</span>
          </div>
          <button aria-label="Close navigation" className="sidebar-close" onClick={() => setMobileOpen(false)} type="button">
            <X size={19} />
          </button>
        </div>
        <div className="sidebar-group-label">Operations</div>
        <nav aria-label="Primary navigation" className="nav-list">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = activeNav.href === item.href;
            return (
              <button
                aria-current={isActive ? 'page' : undefined}
                className={cn('nav-link', isActive && 'nav-link--active')}
                key={item.href}
                onClick={() => goTo(item.href)}
                title={collapsed ? item.label : undefined}
                type="button"
              >
                <Icon size={18} strokeWidth={1.75} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="sidebar-status-dot" />
            <span>Demo services online</span>
          </div>
          <button
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="collapse-sidebar"
            onClick={() => setCollapsed((value) => !value)}
            type="button"
          >
            {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span>Collapse</span></>}
          </button>
        </div>
      </aside>
      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <IconButton className="mobile-menu-button" label="Open navigation" onClick={() => setMobileOpen(true)}>
              <Menu size={20} />
            </IconButton>
            <button aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} className="desktop-menu-button" onClick={() => setCollapsed((value) => !value)} type="button">
              <Menu size={19} />
            </button>
            <div className="search-box">
              <Search size={16} />
              <input
                aria-label="Global search"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && results[0]) goToResult(results[0]);
                }}
                placeholder="Search UAV, engine, flight, alert…"
                value={query}
              />
              {query ? <span className="search-key">ESC</span> : <span className="search-key">⌘ K</span>}
              {query ? (
                <div className="search-results">
                  {results.length ? results.map((result) => (
                    <button key={result.kind + result.title} onClick={() => goToResult(result)} type="button">
                      <span className="search-result-kind">{result.kind}</span>
                      <span><strong>{result.title}</strong><small>{result.subtitle}</small></span>
                    </button>
                  )) : (
                    <div className="search-empty">No matching navigation, UAVs, or alerts.</div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
          <div className="topbar-actions">
            <Badge dot tone={environment === 'Demo' ? 'blue' : 'green'}>{environment}</Badge>
            <div className="notification-wrap">
              <IconButton label="Notifications" onClick={() => setShowNotifications((value) => !value)}>
                <Bell size={18} />
                {openAlerts.length ? <span className="notification-count">{openAlerts.length}</span> : null}
              </IconButton>
              {showNotifications ? (
                <div className="notification-menu">
                  <div className="notification-menu-header">
                    <div><strong>Notifications</strong><span>{openAlerts.length} active operator alerts</span></div>
                    <button onClick={() => goTo('/alerts')} type="button">View all</button>
                  </div>
                  {openAlerts.slice(0, 3).map((alert) => (
                    <button
                      className="notification-item"
                      key={alert.id}
                      onClick={() => {
                        setSelectedUavId(alert.uav_id);
                        setShowNotifications(false);
                        router.push('/alerts');
                      }}
                      type="button"
                    >
                      <Badge tone={alertTone(alert.severity)}>{alert.severity}</Badge>
                      <span><strong>{alert.title}</strong><small>{alert.uav_id} · {alert.id}</small></span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
            <button
              className="user-control"
              onClick={() => toast.info('Operator menu is available in the connected deployment.')}
              type="button"
            >
              <span className="avatar">AO</span>
              <span className="user-copy"><strong>Alex Operator</strong><small>Flight engineering</small></span>
              <ChevronDown size={15} />
            </button>
          </div>
        </header>
        <div className="breadcrumbs">
          <span>Operations</span>
          <ChevronRight size={13} />
          <strong>{activeNav.label}</strong>
          {selectedUavId && ['/live-monitoring', '/engine-health'].includes(activeNav.href) ? (
            <><ChevronRight size={13} /><span>{selectedUavId}</span></>
          ) : null}
        </div>
        <main className="content">{children}</main>
      </div>
      <AddUavWizard />
    </div>
  );
}
