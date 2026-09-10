"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Home, Plane, Database, Radio, Activity, Thermometer, ShieldAlert,
  BarChart3, SlidersHorizontal, Settings, Bell, Search, Menu,
  ChevronDown, User, HelpCircle, PlaneTakeoff, CheckCheck, AlertTriangle, Info
} from "lucide-react";
import { Button } from "@/components/ui/button";

const NAV_ITEMS = [
  { label: "Dashboard", href: "/dashboard", icon: Home },
  { label: "UAV Fleet", href: "/fleet", icon: Plane },
  { label: "Add UAV", href: "/add-uav", icon: PlaneTakeoff },
  { label: "Gateway", href: "/gateway", icon: Database },
  { label: "Live Monitoring", href: "/monitoring", icon: Activity },
  { label: "Engine Health", href: "/health", icon: Thermometer },
  { label: "Alerts", href: "/alerts", icon: ShieldAlert },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
  { label: "Simulator", href: "/simulator", icon: SlidersHorizontal },
  { label: "Settings", href: "/settings", icon: Settings },
];

const BREADCRUMB_MAP: Record<string, string> = {
  dashboard: "Dashboard",
  fleet: "UAV Fleet",
  "add-uav": "Add UAV",
  gateway: "Gateway",
  monitoring: "Live Monitoring",
  health: "Engine Health",
  alerts: "Alerts",
  analytics: "Analytics",
  simulator: "Simulator",
  settings: "Settings",
};

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [unreadNotifications, setUnreadNotifications] = useState(2);
  const activeSegment = pathname.split("/")[1] ?? "";
  const pageTitle = BREADCRUMB_MAP[activeSegment] ?? activeSegment;

  return (
    <div className="ops-app-shell flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={`ops-sidebar flex flex-col border-r border-border bg-sidebar transition-all duration-200 ${mobileNavOpen ? "ops-sidebar-open" : ""} ${
          collapsed ? "w-16" : "w-56"
        } shrink-0`}
      >
        {/* Brand */}
        <div className="flex h-20 items-center gap-2 border-b border-sidebar-border px-3">
          <div className="flex size-11 items-center justify-center rounded-md bg-primary/15 text-primary">
            <PlaneTakeoff className="size-6" />
          </div>
          {!collapsed && (
            <span className="text-base font-bold leading-tight text-foreground">
              UAV Engine<br />Monitoring
            </span>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 py-3">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex h-11 items-center gap-3 rounded-md px-3 text-base transition-colors ${
                  active
                    ? "bg-sidebar-accent font-semibold text-sidebar-accent-foreground border-l-2 border-primary"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                }`}
                title={collapsed ? item.label : undefined}
                onClick={() => setMobileNavOpen(false)}
              >
                <item.icon className="size-5 shrink-0" />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </Link>
            );
          })}
          </nav>
      </aside>
      {mobileNavOpen && <button className="ops-nav-backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}

      {/* Main */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top bar */}
        <header className="ops-topbar flex h-20 shrink-0 items-center gap-3 border-b border-border bg-sidebar px-5">
          <Button
            variant="ghost"
            size="icon-sm"
            className="size-11 text-muted-foreground"
            onClick={() => typeof window !== "undefined" && window.innerWidth < 768 ? setMobileNavOpen(!mobileNavOpen) : setCollapsed(!collapsed)}
          >
            <Menu className="size-6" />
          </Button>
          <div className="flex items-center gap-2 text-base text-muted-foreground">
            <span className="font-semibold text-foreground">{pageTitle}</span>
          </div>
          <label className="ml-3 flex h-11 w-full max-w-md items-center gap-2 rounded-lg border border-border bg-card/90 px-3 text-sm text-muted-foreground shadow-inner shadow-black/10 focus-within:border-primary">
            <Search className="size-4 shrink-0 text-primary" />
            <input className="w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground" placeholder="Search UAV, engine, flight or alert..." aria-label="Global search" />
          </label>
          <div className="ml-auto flex items-center gap-3">
            <div className="relative">
              <button
                className="relative rounded-md p-1 text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                aria-label="Open notifications"
                aria-expanded={notificationsOpen}
                onClick={() => setNotificationsOpen((open) => !open)}
              >
                <Bell className="size-6" />
                {unreadNotifications > 0 && <span className="absolute -right-2 -top-2 flex size-4.5 items-center justify-center rounded-full bg-destructive text-[9px] font-bold text-white">{unreadNotifications}</span>}
              </button>
              {notificationsOpen && <div className="ops-notifications" role="dialog" aria-label="Notifications">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <div><b className="text-sm text-foreground">Notifications</b><p className="mt-0.5 text-[11px] text-muted-foreground">Mission activity and system alerts</p></div>
                  <button className="flex items-center gap-1 text-[11px] font-medium text-primary hover:text-primary/80" onClick={() => setUnreadNotifications(0)}><CheckCheck className="size-3.5" /> Mark all read</button>
                </div>
                <div className="space-y-1 p-2">
                  <Link href="/alerts" onClick={() => { setUnreadNotifications(0); setNotificationsOpen(false); }} className="ops-notification-item"><span className="bg-danger/15 text-danger"><AlertTriangle className="size-4" /></span><div><b>Lubrication degradation detected</b><p>UAV-017 requires attention · 2 min ago</p></div>{unreadNotifications > 0 && <i />}</Link>
                  <Link href="/alerts" onClick={() => { setUnreadNotifications(0); setNotificationsOpen(false); }} className="ops-notification-item"><span className="bg-primary/15 text-primary"><Info className="size-4" /></span><div><b>Telemetry stream restored</b><p>Gateway link is receiving data · 8 min ago</p></div>{unreadNotifications > 0 && <i />}</Link>
                  <Link href="/alerts" onClick={() => setNotificationsOpen(false)} className="ops-notification-item"><span className="bg-success/15 text-success"><CheckCheck className="size-4" /></span><div><b>Mapping validated successfully</b><p>UAV-017 telemetry schema is ready · 21 min ago</p></div></Link>
                </div>
                <Link href="/alerts" onClick={() => setNotificationsOpen(false)} className="block border-t border-border px-4 py-2.5 text-center text-xs font-medium text-primary hover:bg-sidebar-accent">View all alerts</Link>
              </div>}
            </div>
            <button className="text-muted-foreground hover:text-foreground">
              <HelpCircle className="size-5" />
            </button>
            <div className="flex items-center gap-1.5 border-l border-border pl-3">
              <div className="flex size-10 items-center justify-center rounded-full bg-primary/20 text-primary">
                <User className="size-5" />
              </div>
              <div className="hidden text-sm leading-tight md:block">
                <div className="font-medium text-foreground">Mission Control</div>
                <div className="text-muted-foreground">SIH Operator</div>
              </div>
              <ChevronDown className="size-4 text-muted-foreground" />
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="ops-main flex-1 overflow-y-auto p-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 10, filter: "blur(3px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              exit={{ opacity: 0, y: -6, filter: "blur(2px)" }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
