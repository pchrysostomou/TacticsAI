import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'TacticsAI — Real-time Football Analysis',
  description: 'AI-powered tactical analysis: player detection, team classification, bird\'s-eye view, formations, heatmaps and pressing detection.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
