import type { Metadata } from 'next';
import '@/app/globals.css';
import { AppProviders } from '@/components/layout/app-providers';
import { AppShell } from '@/components/layout/app-shell';

export const metadata: Metadata = {
  title: 'AeroSentinel | Engine Operations',
  description: 'UAV engine monitoring operations console',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AppProviders>
          <AppShell>{children}</AppShell>
        </AppProviders>
      </body>
    </html>
  );
}
