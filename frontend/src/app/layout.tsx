import "./globals.css";
import type { Metadata } from "next";
import Navbar from "../components/Navbar";
import { AppWagmiProvider } from "../components/WagmiProvider";
import { Lock } from "lucide-react";

export const metadata: Metadata = {
  title: "Legal Mediverse",
  description: "Asynchronous Multi-Agent Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-col selection:bg-cyan-500 selection:text-slate-950">
        
        {/* Hero Ambient Glow Layer */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
          <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[1000px] h-[600px] rounded-full bg-gradient-to-tr from-cyan-500/15 via-indigo-500/10 to-violet-500/5 blur-[140px]" />
          <div className="absolute top-[40%] right-[-10%] w-[500px] h-[500px] rounded-full bg-gradient-to-bl from-teal-500/10 to-indigo-500/5 blur-[120px]" />
        </div>

        <AppWagmiProvider>
          <Navbar />

          <main className="z-10 relative flex-1 flex flex-col w-full">
            {children}
          </main>
        </AppWagmiProvider>

        {/* Global Footer */}
        <footer className="w-full border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md py-8 z-10 relative mt-auto">
          <div className="max-w-7xl mx-auto px-6 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <Lock className="w-3.5 h-3.5 text-emerald-400"/>
              <span className="font-medium">Presidio PHI Scrubbing Active • End-to-End Encrypted</span>
            </div>
            <p className="text-slate-400 font-medium">© 2026 Legal Mediverse. Built for clinical, legal, and educational advocacy.</p>
          </div>
        </footer>

      </body>
    </html>
  );
}
