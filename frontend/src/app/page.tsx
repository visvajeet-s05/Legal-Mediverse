import Link from "next/link";
import { 
  Stethoscope, 
  GraduationCap, 
  ShieldCheck, 
  Scale, 
  ArrowRight, 
  Bot, 
  FileCheck 
} from "lucide-react";

export default function Home() {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 sm:px-8 py-16 sm:py-24 flex flex-col items-center justify-center text-center space-y-20">
      
      {/* Hero Section */}
      <div className="max-w-4xl space-y-8 flex flex-col items-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/90 border border-cyan-500/30 text-cyan-400 text-xs font-semibold tracking-wider uppercase shadow-inner shadow-cyan-500/10">
          <Bot className="w-4 h-4 text-cyan-400 animate-pulse"/>
          <span>Multi-Agent System • Medical • EdTech • Web3 Escrow</span>
        </div>

        <h1 className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.1]">
          The Asynchronous Platform for{" "}
          <span className="bg-gradient-to-r from-cyan-400 via-teal-300 to-indigo-400 bg-clip-text text-transparent">
            Clinical, Educational, & Legal Advocacy
          </span>
        </h1>

        <p className="text-base sm:text-xl text-slate-300 leading-relaxed max-w-3xl font-normal">
          Connecting AI clinical triage, interactive active-recall graphs, smart contract escrows, and insurance appeal contract redliners into a cohesive, zero-trust ecosystem.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4 w-full sm:w-auto">
          <Link className="w-full sm:w-auto px-8 py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-cyan-500/25 hover:scale-[1.02] active:scale-[0.98]" href="/health">
            <span>Launch Health Dashboard</span>
            <ArrowRight className="w-4 h-4"/>
          </Link>
          
          <Link className="w-full sm:w-auto px-8 py-4 bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-slate-600 rounded-xl flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]" href="/docs">
            <FileCheck className="w-4 h-4 text-slate-400"/>
            <span>Explore Documentation</span>
          </Link>
        </div>
      </div>

      {/* Feature Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full text-left">
        
        <div className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-8 hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all duration-300 flex flex-col justify-between space-y-8 shadow-sm hover:shadow-cyan-500/10">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
              <Stethoscope className="w-6 h-6"/>
            </div>
            <h3 className="text-2xl font-bold text-white group-hover:text-cyan-400 transition-colors">Health & Medical Triage</h3>
            <p className="text-slate-400 text-base leading-relaxed">
              Clinical RAG analysis of symptoms using PubMed abstracts and ICD-10 medical codings. Powered by Gemini 1.5 Flash vision and featuring a binary Cornerstone.js DICOM image viewer.
            </p>
          </div>
          <Link className="inline-flex items-center gap-2 text-sm font-bold text-cyan-400 hover:text-cyan-300 tracking-wide uppercase" href="/health">
            <span>Open Medical Dashboard</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform"/>
          </Link>
        </div>

        <div className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-8 hover:border-indigo-500/50 hover:bg-slate-900/80 transition-all duration-300 flex flex-col justify-between space-y-8 shadow-sm hover:shadow-indigo-500/10">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
              <GraduationCap className="w-6 h-6"/>
            </div>
            <h3 className="text-2xl font-bold text-white group-hover:text-indigo-400 transition-colors">Interactive EdTech Studio</h3>
            <p className="text-slate-400 text-base leading-relaxed">
              Automatically convert uploaded medical notes into visual concept graphs and Anki study flashcards. Generate multi-host podcast transcripts synced with real-time waveform highlight players.
            </p>
          </div>
          <Link className="inline-flex items-center gap-2 text-sm font-bold text-indigo-400 hover:text-indigo-300 tracking-wide uppercase" href="/edu">
            <span>Open Study Center</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform"/>
          </Link>
        </div>

        <div className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-8 hover:border-emerald-500/50 hover:bg-slate-900/80 transition-all duration-300 flex flex-col justify-between space-y-8 shadow-sm hover:shadow-emerald-500/10">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
              <ShieldCheck className="w-6 h-6"/>
            </div>
            <h3 className="text-2xl font-bold text-white group-hover:text-emerald-400 transition-colors">Verified Crowdfunding Escrow</h3>
            <p className="text-slate-400 text-base leading-relaxed">
              Campaign verification via OCR billing receipt validation. Funds are locked inside Solidity smart contract escrows on Polygon and released to verified medical providers.
            </p>
          </div>
          <Link className="inline-flex items-center gap-2 text-sm font-bold text-emerald-400 hover:text-emerald-300 tracking-wide uppercase" href="/community">
            <span>View Escrow Campaigns</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform"/>
          </Link>
        </div>

        <div className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-8 hover:border-amber-500/50 hover:bg-slate-900/80 transition-all duration-300 flex flex-col justify-between space-y-8 shadow-sm hover:shadow-amber-500/10">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 group-hover:scale-110 transition-transform">
              <Scale className="w-6 h-6"/>
            </div>
            <h3 className="text-2xl font-bold text-white group-hover:text-amber-400 transition-colors">Legal Advocacy & Redlining</h3>
            <p className="text-slate-400 text-base leading-relaxed">
              Draft formal appeal letters citing ACA Section 2719, analyze hospital contracts side-by-side to flag predatory clauses, and generate one-click HIPAA release requests.
            </p>
          </div>
          <Link className="inline-flex items-center gap-2 text-sm font-bold text-amber-400 hover:text-amber-300 tracking-wide uppercase" href="/law">
            <span>Open Legal Desk</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform"/>
          </Link>
        </div>

      </div>

    </div>
  );
}
