"use client";

import React, { useState, useEffect } from "react";
import { 
  Play, 
  Pause, 
  FileText, 
  Brain, 
  UploadCloud, 
  Loader2, 
  Download, 
  Sparkles, 
  Radio, 
  Layers,
  CheckCircle2
} from "lucide-react";

export default function EduPage() {
  // Form Uploader State
  const [title, setTitle] = useState("");
  const [notesText, setNotesText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Dynamic Package State
  const [packageData, setPackageData] = useState<any>(null);

  // Audio Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  // Active Flashcard Flip States
  const [flippedCards, setFlippedCards] = useState<{ [key: number]: boolean }>({});

  // Audio playback ticker effect
  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentTime((prev) => {
          if (prev >= 24) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 0.5;
        });
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleStudySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!notesText && !title) return;
    setIsGenerating(true);

    const formData = new FormData();
    formData.append("title", title || "Medical Study Notes");
    formData.append("content", notesText);
    if (file) {
      formData.append("file", file);
    }

    try {
      const res = await fetch("/api/v1/edu/recall-engine", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      // Dynamically generate script tailored to topic
      const dynamicTopic = title || "Clinical Principles";
      const dynamicTranscript = [
        { host: "Dr. Alex", text: `Welcome to Smart Study Podcast. Today we are diving into ${dynamicTopic}.`, start: 0.0, end: 5.5 },
        { host: "Dr. Sam", text: `Thanks Dr. Alex. Let's analyze the core concepts around ${dynamicTopic} and key study notes.`, start: 6.0, end: 11.5 },
        { host: "Dr. Alex", text: `Reviewing user notes: "${notesText.slice(0, 80)}..."`, start: 12.0, end: 17.5 },
        { host: "Dr. Sam", text: `Use the Visual Concept Map and interactive flashcards below to test your active recall!`, start: 18.0, end: 23.5 }
      ];

      setPackageData({
        title: dynamicTopic,
        transcript: dynamicTranscript,
        graph: data.react_flow_graph,
        flashcards: data.flashcards || []
      });
    } catch (err) {
      console.error(err);
      alert("Failed to generate study package.");
    } finally {
      setIsGenerating(false);
    }
  };

  const toggleCardFlip = (idx: number) => {
    setFlippedCards((prev) => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  // Fallback state if package has not been generated yet
  const activeTitle = packageData?.title || title || "Pathophysiology of Peripheral Edema";
  const activeTranscript = packageData?.transcript || [
    { host: "Dr. Alex", text: "Welcome to Smart Study Podcast. Today we're breaking down critical clinical pathophysiology.", start: 0.0, end: 5.5 },
    { host: "Dr. Sam", text: "Thanks Dr. Alex. Let me review vascular permeability and tissue fluid retention dynamics.", start: 6.0, end: 11.5 },
    { host: "Dr. Alex", text: "When pressure increases, fluid shifts into interstitial spaces causing visible edema upon waking.", start: 12.0, end: 17.5 },
    { host: "Dr. Sam", text: "Review the Visual Concept Map and interactive flipcards below to lock in these core medical concepts.", start: 18.0, end: 23.5 }
  ];
  const activeFlashcards = packageData?.flashcards || [
    {
      question: "What primary mechanism drives dependent lower extremity edema upon waking?",
      answer: "Postural fluid accumulation, elevated venous hydrostatic pressure, and localized tissue permeability."
    },
    {
      question: "What initial non-pharmacological interventions are recommended for edema?",
      answer: "Leg elevation above heart level during rest, sodium restriction, and fluid intake monitoring."
    }
  ];

  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-8 flex flex-col gap-8">
      
      {/* Page Title Header */}
      <div className="flex flex-col gap-2 border-b border-slate-800 pb-6">
        <div className="flex items-center gap-2.5 text-indigo-400">
          <Brain className="w-6 h-6 animate-pulse" />
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Smart Study Studio</h1>
        </div>
        <p className="text-slate-400 text-sm max-w-3xl">
          Transform raw medical notes, transcripts, or PDFs into AI podcast audio conversations, interactive visual concept maps, and Anki-ready study flashcards.
        </p>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column (5 Cols): Note/File Uploader Form */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-md">
            <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
              <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded-lg text-indigo-400">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Upload Study Notes</h2>
                <p className="text-zinc-500 text-xs mt-0.5">Medical notes, PDFs, or lecture transcripts</p>
              </div>
            </div>

            <form onSubmit={handleStudySubmit} className="flex flex-col gap-5">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">Topic / Module Title</label>
                <input
                  type="text"
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="E.g., Pathophysiology of Peripheral Edema"
                  className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">Paste Clinical Notes</label>
                <textarea
                  required
                  value={notesText}
                  onChange={(e) => setNotesText(e.target.value)}
                  placeholder="Paste lecture notes or textbook excerpt here..."
                  rows={5}
                  className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 transition-colors resize-none"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">Attach Document (PDF/TXT Optional)</label>
                <div className="flex items-center justify-center border border-dashed border-slate-800 rounded-xl p-5 bg-slate-950/40 hover:bg-slate-950/80 transition-colors cursor-pointer relative">
                  <input
                    type="file"
                    accept=".pdf,.txt,.docx"
                    onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="text-center flex flex-col items-center gap-1.5">
                    <UploadCloud className="text-slate-500 w-6 h-6" />
                    <span className="text-xs font-medium text-slate-400">
                      {file ? file.name : "Click or drag to attach PDF or TXT document"}
                    </span>
                    <span className="text-[10px] text-slate-600">Max file size 10MB</span>
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={isGenerating}
                className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900/50 text-white font-bold rounded-xl text-sm flex items-center justify-center gap-2 transition-all shadow-md shadow-indigo-600/10"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Building Study Package...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Study Package</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column (7 Cols): Podcast, Visual Graph & Instant Flashcards */}
        <div className="lg:col-span-7 flex flex-col gap-8">
          
          {/* AI Audio Podcast Generator Panel */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-md">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-violet-500/10 border border-violet-500/30 rounded-lg text-violet-400">
                  <Radio className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">AI Audio Podcast Generator</h2>
                  <p className="text-zinc-500 text-xs mt-0.5">Dual AI-host conversational review format</p>
                </div>
              </div>
              <span className="text-[10px] font-mono font-bold bg-violet-500/10 text-violet-400 border border-violet-500/20 px-2.5 py-1 rounded-full">
                HD AUDIO
              </span>
            </div>

            {/* Audio Controls Container */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 flex flex-col gap-5">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="w-12 h-12 rounded-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white flex items-center justify-center shadow-lg shadow-indigo-500/10 transition-all shrink-0"
                >
                  {isPlaying ? <Pause className="w-5 h-5 fill-white" /> : <Play className="w-5 h-5 fill-white ml-0.5" />}
                </button>
                
                <div className="flex-1 min-w-0">
                  <h4 className="text-xs font-bold text-white truncate">{activeTitle}</h4>
                  <p className="text-[10px] text-slate-400 mt-0.5">Dual-Host: Dr. Alex & Dr. Sam</p>
                </div>

                <span className="text-xs font-mono text-slate-400 shrink-0">
                  00:{currentTime.toFixed(0).padStart(2, "0")} / 00:24
                </span>
              </div>

              {/* Animated Waveform Bars */}
              <div className="h-10 flex items-end gap-1 px-1 border-t border-slate-800/80 pt-3">
                {Array.from({ length: 44 }).map((_, idx) => {
                  const isActive = currentTime * 1.8 >= idx;
                  const height = isPlaying 
                    ? Math.sin(idx * 0.45 + currentTime * 2) * 16 + 22
                    : Math.sin(idx * 0.45) * 10 + 16;
                  return (
                    <div
                      key={idx}
                      className={`flex-1 rounded-full transition-all duration-200 ${
                        isActive ? "bg-indigo-500" : "bg-slate-800"
                      }`}
                      style={{ height: `${height}px` }}
                    />
                  );
                })}
              </div>
            </div>

            {/* Interactive Dual-Speaker Transcript */}
            <div className="flex flex-col gap-3 max-h-[220px] overflow-y-auto pr-1">
              {activeTranscript.map((line: any, idx: number) => {
                const isActiveLine = currentTime >= line.start && currentTime <= line.end;
                return (
                  <div
                    key={idx}
                    onClick={() => setCurrentTime(line.start)}
                    className={`p-3 rounded-xl border transition-all cursor-pointer ${
                      isActiveLine 
                        ? "bg-indigo-950/30 border-indigo-500/40 shadow-sm" 
                        : "bg-slate-950/40 border-slate-800/60 opacity-70 hover:opacity-100"
                    }`}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className={`text-[10px] font-extrabold uppercase tracking-wider ${
                        line.host === "Dr. Alex" ? "text-violet-400" : "text-emerald-400"
                      }`}>
                        {line.host}
                      </span>
                      <span className="text-[9px] text-slate-500 font-mono">{line.start.toFixed(1)}s</span>
                    </div>
                    <p className="text-xs text-slate-200 leading-normal">{line.text}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Visual Concept Map Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col gap-4 shadow-md">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Visual Concept Map</h2>
                  <p className="text-zinc-500 text-xs mt-0.5">Interactive node breakdown of clinical concepts</p>
                </div>
              </div>
            </div>

            {/* Node Graph Container */}
            <div className="h-64 border border-slate-800 bg-slate-950 rounded-xl relative overflow-hidden flex items-center justify-center p-4">
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <line x1="120" y1="128" x2="300" y2="70" stroke="rgba(99, 102, 241, 0.4)" strokeWidth="2" strokeDasharray="4" />
                <line x1="120" y1="128" x2="300" y2="186" stroke="rgba(99, 102, 241, 0.4)" strokeWidth="2" strokeDasharray="4" />
                <line x1="300" y1="70" x2="480" y2="128" stroke="rgba(16, 185, 129, 0.4)" strokeWidth="2" />
                <line x1="300" y1="186" x2="480" y2="128" stroke="rgba(16, 185, 129, 0.4)" strokeWidth="2" />
              </svg>

              {/* Node Cards */}
              <div className="absolute left-[30px] top-[108px] px-3.5 py-2.5 bg-indigo-950/80 border border-indigo-500/50 rounded-xl text-xs font-bold text-indigo-300 shadow-lg max-w-[150px] truncate">
                {activeTitle}
              </div>

              <div className="absolute left-[220px] top-[48px] px-3.5 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-xs font-semibold text-slate-200 shadow-md max-w-[150px] truncate">
                {packageData?.graph?.nodes[1]?.data?.label || "Etiology & Triggers"}
              </div>

              <div className="absolute left-[220px] top-[166px] px-3.5 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-xs font-semibold text-slate-200 shadow-md max-w-[150px] truncate">
                {packageData?.graph?.nodes[2]?.data?.label || "Clinical Presentation"}
              </div>

              <div className="absolute right-[30px] top-[108px] px-3.5 py-2.5 bg-emerald-950/80 border border-emerald-500/50 rounded-xl text-xs font-bold text-emerald-300 shadow-lg max-w-[150px] truncate">
                Protocols & Action
              </div>
            </div>
          </div>

          {/* Instant Study Flashcards Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col gap-5 shadow-md">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
                  <Brain className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Instant Study Flashcards</h2>
                  <p className="text-zinc-500 text-xs mt-0.5">Click card to flip and test active recall</p>
                </div>
              </div>
              <button 
                onClick={() => alert("Exporting Anki package (.apkg)...")}
                className="flex items-center gap-1.5 text-xs font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-3 py-1.5 rounded-lg hover:bg-cyan-500/20 transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export to Anki (.apkg)</span>
              </button>
            </div>

            {/* Flashcard Flip Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeFlashcards.map((card: any, idx: number) => {
                const isFlipped = flippedCards[idx];
                return (
                  <div
                    key={idx}
                    onClick={() => toggleCardFlip(idx)}
                    className="h-40 bg-slate-950 border border-slate-800 hover:border-indigo-500/40 rounded-xl p-5 flex flex-col justify-between cursor-pointer transition-all shadow-md relative overflow-hidden group"
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-extrabold uppercase tracking-widest text-indigo-400">
                        {isFlipped ? "ANSWER" : "QUESTION"}
                      </span>
                      <span className="text-[10px] text-slate-500">Click to flip</span>
                    </div>

                    <p className={`text-xs leading-relaxed transition-all ${isFlipped ? "text-emerald-300 font-medium" : "text-slate-200"}`}>
                      {isFlipped ? (card.answer || card.back) : (card.question || card.front)}
                    </p>

                    <div className="flex items-center gap-1 text-[10px] text-slate-500">
                      <CheckCircle2 className="w-3 h-3 text-indigo-400" />
                      <span>{isFlipped ? "Recall Checked" : "Test Your Memory"}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
