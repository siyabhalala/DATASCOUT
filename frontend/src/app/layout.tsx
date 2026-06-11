import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'DataScout — AI Dataset Research Agent',
  description: 'Intelligent dataset discovery with deterministic evaluation and grounded AI explanations.',
  icons: {
    icon: '/logo.png',
    apple: '/logo.png',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Jost:wght@300;400;500;600&display=swap"
          rel="stylesheet"
        />
        {/*
          ── LOGO INSTRUCTIONS ──────────────────────────────────────────
          Place your startup logo file at:
            datascout/frontend/public/logo.png   ← main logo (PNG recommended)
            datascout/frontend/public/logo.svg   ← optional SVG version

          The favicon is automatically read from /public/logo.png via the
          metadata.icons config above. No extra steps needed.

          Recommended sizes:
            Logo PNG : at least 256×256px, transparent background preferred
            Favicon  : 32×32 or 64×64 (Next.js resizes automatically)

          If you have a .ico favicon, also place it at:
            datascout/frontend/public/favicon.ico
          ──────────────────────────────────────────────────────────────
        */}
      </head>
      <body>{children}</body>
    </html>
  )
}
