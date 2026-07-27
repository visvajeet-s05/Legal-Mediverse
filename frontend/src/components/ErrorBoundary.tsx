"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
          <div className="max-w-md w-full bg-slate-900/80 border border-rose-500/20 rounded-3xl p-8 flex flex-col items-center gap-6 text-center shadow-2xl">
            {/* Icon */}
            <div className="w-16 h-16 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center">
              <AlertTriangle className="w-8 h-8 text-rose-400" />
            </div>

            {/* Content */}
            <div className="flex flex-col gap-2">
              <h2 className="text-xl font-bold text-white">Something went wrong</h2>
              <p className="text-sm text-slate-400 leading-relaxed">
                An unexpected error occurred. Please try refreshing the page or
                return to the home page.
              </p>
              {process.env.NODE_ENV === "development" && this.state.error && (
                <div className="mt-2 p-3 bg-slate-950 rounded-xl border border-slate-800 text-left">
                  <p className="text-[10px] font-mono text-rose-400 font-bold mb-1">
                    {this.state.error.name}
                  </p>
                  <p className="text-[10px] font-mono text-slate-400 break-all">
                    {this.state.error.message}
                  </p>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-3 w-full">
              <button
                onClick={this.handleReset}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Try Again
              </button>
              <button
                onClick={() => (window.location.href = "/")}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-bold rounded-xl text-xs transition-colors"
              >
                <Home className="w-4 h-4" />
                Go Home
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

