"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, Wallet, LogOut, User, Menu, X, Loader2 } from "lucide-react";
import { useAccount, useConnect, useDisconnect } from "wagmi";
import { injected } from "wagmi/connectors";

const NAV_LINKS = [
  { href: "/health", label: "Clinical Triage" },
  { href: "/edu", label: "Audio & Recall" },
  { href: "/community", label: "Escrow Campaigns" },
  { href: "/law", label: "Legal & Compliance" },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { address, isConnected } = useAccount();
  const { connect, isPending } = useConnect();
  const { disconnect } = useDisconnect();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setAuthToken(sessionStorage.getItem("access_token"));
  }, []);

  const handleConnect = async () => {
    if (isConnected) {
      disconnect();
    } else {
      connect({ connector: injected() });
    }
  };

  if (!mounted) return null;

  return (
    <header className="w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <Link href="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
          <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <Activity className="w-5 h-5 text-cyan-400"/>
          </div>
          <span className="font-bold text-lg tracking-wide text-white max-sm:hidden">
            LEGAL MEDIVERSE
          </span>
          <span className="font-bold text-lg tracking-wide text-white sm:hidden">LM</span>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6 text-xs font-semibold uppercase tracking-wider text-slate-400">
          {NAV_LINKS.map((link) => {
            const isActive = pathname?.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`hover:text-cyan-400 transition-colors ${
                  isActive ? "text-cyan-400" : "text-slate-400"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Right Side: Wallet + Auth */}
        <div className="flex items-center gap-3">
          {/* Auth buttons (desktop) */}
          <div className="hidden md:flex items-center gap-2">
            {authToken ? (
              <button
                onClick={() => {
                  sessionStorage.removeItem("access_token");
                  sessionStorage.removeItem("user_role");
                  sessionStorage.removeItem("user_id");
                  setAuthToken(null);
                  router.push("/");
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-colors"
              >
                <LogOut className="w-3 h-3" />
                Logout
              </button>
            ) : (
              <Link
                href="/auth/login"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-400 border border-cyan-500/30 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-colors"
              >
                <User className="w-3 h-3" />
                Sign In
              </Link>
            )}
          </div>

          {/* Connect Wallet Button */}
          <button
            onClick={handleConnect}
            disabled={isPending}
            className="flex items-center gap-2 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-cyan-500/50 rounded-lg text-xs font-bold uppercase tracking-wider transition-all disabled:opacity-60"
          >
            {isPending ? (
              <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
            ) : (
              <Wallet className="w-4 h-4 text-cyan-400" />
            )}
            <span className="max-sm:hidden">
              {isConnected ? `${address?.slice(0, 6)}...${address?.slice(-4)}` : "Connect Wallet"}
            </span>
          </button>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 text-slate-400 hover:text-white"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {mobileOpen && (
        <div className="md:hidden border-t border-slate-800/80 bg-slate-950/95 backdrop-blur-md">
          <div className="px-6 py-4 flex flex-col gap-3">
            {NAV_LINKS.map((link) => {
              const isActive = pathname?.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className={`text-sm font-semibold uppercase tracking-wider py-2 transition-colors ${
                    isActive ? "text-cyan-400" : "text-slate-400"
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
            <div className="border-t border-slate-800 pt-3 mt-1 flex flex-col gap-2">
              {authToken ? (
                <button
                  onClick={() => {
                    sessionStorage.removeItem("access_token");
                    setAuthToken(null);
                    setMobileOpen(false);
                    router.push("/");
                  }}
                  className="text-sm text-rose-400 font-semibold py-2"
                >
                  Logout
                </button>
              ) : (
                <Link
                  href="/auth/login"
                  onClick={() => setMobileOpen(false)}
                  className="text-sm text-cyan-400 font-semibold py-2"
                >
                  Sign In
                </Link>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

