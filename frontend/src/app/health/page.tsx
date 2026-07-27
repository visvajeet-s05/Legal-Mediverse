"use client";

import React, { useState } from "react";
import { 
  ShieldAlert, 
  Image as ImageIcon, 
  Activity, 
  FileText, 
  Check, 
  Loader2, 
  CheckCircle,
  HelpCircle,
  Database,
  Flame,
  Moon,
  Footprints
} from "lucide-react";
import DicomViewer from "../../components/DicomViewer";

export default function HealthPage() {
  // Safety Intercept Modal State
  const [modalApproved, setModalApproved] = useState(false);
  const [showModal, setShowModal] = useState(true);

  // Form State
  const [description, setDescription] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [dicomCanvasFrame, setDicomCanvasFrame] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Daily Health Tracker State
  const [logType, setLogType] = useState("steps");
  const [logValue, setLogValue] = useState("");
  const [isLogging, setIsLogging] = useState(false);
  const [fhirLogToast, setFhirLogToast] = useState<{ id: string; timestamp: string } | null>(null);

  const handleTriageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description) return;
    setIsSubmitting(true);
    setResult(null);

    const formData = new FormData();
    formData.append("description", description);
    if (imageFile) {
      formData.append("image", imageFile);
    } else if (dicomCanvasFrame) {
      // Convert base64 data URL to Blob for FormData submission
      try {
        const fetchRes = await fetch(dicomCanvasFrame);
        const blob = await fetchRes.blob();
        formData.append("image", blob, "dicom_rendered_frame.png");
      } catch (err) {
        console.error("Failed to attach DICOM canvas frame:", err);
      }
    }

    try {
      const res = await fetch("/api/v1/health/triage", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Triage request failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLogFhirObservation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!logValue) return;
    setIsLogging(true);
    setFhirLogToast(null);

    const val = parseFloat(logValue);

    let codeDetails = {
      system: "http://loinc.org",
      code: "55423-8",
      display: "Number of steps in unspecified time Pedometer"
    };
    let unitStr = "steps";

    if (logType === "sleep") {
      codeDetails = {
        system: "http://loinc.org",
        code: "9318-7",
        display: "Total sleep duration"
      };
      unitStr = "hours";
    } else if (logType === "nutrition") {
      codeDetails = {
        system: "http://loinc.org",
        code: "9052-2",
        display: "Calorie intake Total"
      };
      unitStr = "kcal";
    }

    const fhirPayload = {
      resourceType: "Observation",
      status: "final",
      category: [
        {
          coding: [
            {
              system: "http://terminology.hl7.org/CodeSystem/observation-category",
              code: "vital-signs"
            }
          ]
        }
      ],
      code: {
        coding: [codeDetails]
      },
      valueQuantity: {
        value: val,
        unit: unitStr
      }
    };

    try {
      const res = await fetch("/api/v1/health/fhir/observation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fhirPayload)
      });
      const data = await res.json();
      if (res.ok) {
        setFhirLogToast({
          id: data.observation_id,
          timestamp: data.timestamp
        });
        setLogValue("");
      } else {
        alert(data.detail || "Error logging FHIR observation.");
      }
    } catch (err) {
      alert("Connection failed while logging FHIR observation.");
    } finally {
      setIsLogging(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-8 flex flex-col gap-8">
      
      {/* Safety Intercept Modal */}
      {showModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-rose-500/20 rounded-2xl max-w-lg w-full p-8 flex flex-col gap-6 text-center shadow-2xl">
            <div className="w-16 h-16 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center mx-auto text-rose-400">
              <ShieldAlert className="w-8 h-8" />
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-2xl font-bold text-white tracking-tight">Clinical Safety Disclaimer</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                You are initiating an AI-powered diagnostic and image triage service. All outcomes, confidence indexes, and ICD-10 suggestions are for advocacy and review support only. If you are experiencing a medical emergency, please call emergency services immediately.
              </p>
            </div>
            <button
              onClick={() => {
                setModalApproved(true);
                setShowModal(false);
              }}
              className="w-full py-3 bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white font-semibold rounded-xl text-sm transition-all"
            >
              I Understand and Agree
            </button>
          </div>
        </div>
      )}

      {/* Page Title */}
      <div className="flex flex-col gap-2 border-b border-slate-800 pb-6">
        <div className="flex items-center gap-2.5 text-cyan-400">
          <Activity className="w-6 h-6 animate-pulse" />
          <h1 className="text-3xl font-extrabold text-white tracking-tight">AI Health Check</h1>
        </div>
        <p className="text-slate-400 text-sm max-w-3xl">
          Describe symptoms, upload medical scans, and securely track your daily health markers in our compliance-first clinical triage workspace.
        </p>
      </div>

      {/* Main Page Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        
        {/* Left Columns (Span 2): AI Health Check Form & Medical Scan Viewer */}
        <div className="lg:col-span-2 flex flex-col gap-8">
          
          {/* AI Health Check Form Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6 shadow-md">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
                  <Activity className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">AI Health Check</h2>
                  <p className="text-zinc-500 text-xs mt-0.5">Scrubbed via Presidio for end-to-end patient privacy</p>
                </div>
              </div>
              {result && result.phi_elements_scrubbed_count !== undefined && (
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[11px] font-bold">
                  <CheckCircle className="w-3.5 h-3.5" />
                  <span>✓ Presidio Active: {result.phi_elements_scrubbed_count} PHI elements scrubbed</span>
                </div>
              )}
            </div>

            <form onSubmit={handleTriageSubmit} className="flex flex-col gap-6">
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">Describe Symptoms or Injury</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="E.g., After waking up in the morning, my legs swell. What is that?"
                  rows={4}
                  required
                  className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500 transition-colors resize-none"
                />
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">Upload Injury Image (Optional)</label>
                <div className="flex items-center justify-center border border-dashed border-slate-800 rounded-xl p-6 bg-slate-950/40 hover:bg-slate-950/80 transition-colors cursor-pointer relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setImageFile(e.target.files ? e.target.files[0] : null)}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="text-center flex flex-col items-center gap-2">
                    <ImageIcon className="text-slate-500 w-8 h-8" />
                    <span className="text-xs font-medium text-slate-400">
                      {imageFile ? imageFile.name : "Click or drag to upload JPG/PNG image"}
                    </span>
                    <span className="text-[10px] text-slate-600">Max size 5MB</span>
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-3.5 bg-cyan-500 hover:bg-cyan-400 disabled:bg-cyan-900/50 text-slate-950 font-bold rounded-xl text-sm flex items-center justify-center gap-2 transition-all shadow-md shadow-cyan-500/10"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Processing Clinical RAG & Gemini Vision...</span>
                  </>
                ) : (
                  <span>Analyze & Triage Injury</span>
                )}
              </button>
            </form>
          </div>

          {/* Medical Scan Viewer Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6 shadow-md">
            <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
              <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded-lg text-indigo-400">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Medical Scan Viewer (.DCM)</h2>
                <p className="text-zinc-500 text-xs mt-0.5">Cornerstone binary browser DICOM parser & Gemini Vision</p>
              </div>
            </div>
            <DicomViewer 
              onDicomAnalyzed={(parsedData, canvasUrl) => {
                setDescription(`[DICOM Scan Loaded] Patient ID: ${parsedData.patientId}, Modality: ${parsedData.modality}, Study Date: ${parsedData.studyDate}`);
                if (canvasUrl) {
                  setDicomCanvasFrame(canvasUrl);
                }
              }} 
            />
          </div>

        </div>

        {/* Right Column: Diagnostic Reports & Daily Health Tracker */}
        <div className="flex flex-col gap-8">
          
          {/* Skeleton Loaders during Submission */}
          {isSubmitting && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6 shadow-xl animate-pulse">
              <div className="flex justify-between items-center border-b border-slate-800 pb-4">
                <div className="h-6 w-32 bg-slate-800 rounded-lg"></div>
                <div className="h-5 w-20 bg-slate-800 rounded-full"></div>
              </div>
              <div className="flex flex-col gap-2">
                <div className="h-4 w-28 bg-slate-800 rounded"></div>
                <div className="h-20 w-full bg-slate-800 rounded-xl"></div>
              </div>
              <div className="flex flex-col gap-3">
                <div className="h-4 w-32 bg-slate-800 rounded"></div>
                <div className="h-24 w-full bg-slate-800 rounded-xl"></div>
              </div>
            </div>
          )}

          {/* Triage Results Report Card */}
          {result && !isSubmitting && (
            <div className="bg-slate-900/60 border border-cyan-500/20 bg-cyan-950/5 rounded-2xl p-8 flex flex-col gap-6 shadow-xl animate-in fade-in duration-300">
              
              {/* Emergency Banner Alert */}
              {(result.risk_level === "Urgent" || result.severity === "critical" || result.severity === "severe") && (
                <div className="bg-rose-950/70 border border-rose-500/50 p-4 rounded-xl flex gap-3 text-rose-200 animate-bounce">
                  <ShieldAlert className="text-rose-400 w-6 h-6 shrink-0 mt-0.5" />
                  <div className="flex flex-col gap-1 text-xs">
                    <span className="font-extrabold uppercase tracking-wider text-rose-300">⚠️ IMMEDIATE ACTION REQUIRED</span>
                    <p className="leading-relaxed">
                      Your described symptoms may indicate an acute medical emergency. Please call emergency services (911/112) or proceed to the nearest emergency department immediately.
                    </p>
                  </div>
                </div>
              )}

              <div className="flex justify-between items-center border-b border-slate-800 pb-4">
                <h2 className="text-lg font-bold text-white">Diagnostic Report</h2>
                <span className={`text-[10px] uppercase font-bold tracking-widest px-3 py-1 rounded-full ${
                  (result.risk_level === "Urgent" || result.severity === "critical" || result.severity === "severe") 
                    ? "bg-rose-500/10 text-rose-400 border border-rose-500/30" 
                    : (result.risk_level === "Moderate" || result.severity === "moderate")
                    ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                    : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                }`}>
                  {result.risk_level ? `${result.risk_level} Risk` : (result.severity ? `${result.severity.toUpperCase()} Risk` : "Low Risk")}
                </span>
              </div>

              {/* Assessment Description */}
              <div className="flex flex-col gap-2">
                <span className="text-[10px] font-bold tracking-wider text-zinc-400 uppercase">Assessment Summary</span>
                <p className="text-slate-300 text-xs leading-relaxed bg-black/40 p-4 rounded-xl border border-slate-800">
                  {result.summary || result.recommended_immediate_treatment || result.treatment || "No clinical summary returned."}
                </p>
              </div>

              {/* Differential Diagnoses */}
              {((result.diagnoses && result.diagnoses.length > 0) || (result.differential_diagnoses && result.differential_diagnoses.length > 0)) && (
                <div className="flex flex-col gap-3">
                  <span className="text-[10px] font-bold tracking-wider text-zinc-400 uppercase">Differential Diagnoses</span>
                  <div className="flex flex-col gap-2">
                    {(result.diagnoses || []).map((diag: any, i: number) => (
                      <div key={i} className="bg-black/35 border border-slate-800 rounded-xl p-4 flex flex-col gap-2">
                        <div className="flex justify-between items-start">
                          <h4 className="font-bold text-white text-xs">{diag.condition}</h4>
                          <span className="text-[10px] font-bold text-cyan-400">{diag.match_percentage || `${diag.confidence_score}%`}</span>
                        </div>
                        <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
                          <span>ICD-10: {diag.icd10_code}</span>
                          <span>Source: {diag.source || diag.citation || "Medical Guideline Standard"}</span>
                        </div>
                        <p className="text-[11px] text-zinc-400 leading-normal">{diag.description || diag.reasoning}</p>
                      </div>
                    ))}
                    {!result.diagnoses && (result.differential_diagnoses || []).map((diag: any, i: number) => (
                      <div key={i} className="bg-black/35 border border-slate-800 rounded-xl p-4 flex flex-col gap-2">
                        <div className="flex justify-between items-start">
                          <h4 className="font-bold text-white text-xs">{diag.condition}</h4>
                          <span className="text-[10px] font-bold text-cyan-400">{diag.match_percentage || `${diag.confidence_score}% Match`}</span>
                        </div>
                        <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
                          <span>ICD-10: {diag.icd10_code}</span>
                          <span>Source: {diag.citation || "PubMed"}</span>
                        </div>
                        <p className="text-[11px] text-zinc-400 leading-normal">{diag.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Legal/Appeal Flag */}
              {result.requires_appeal && (
                <div className="bg-rose-500/5 border border-rose-500/25 p-4 rounded-xl flex gap-3">
                  <ShieldAlert className="text-rose-400 w-5 h-5 shrink-0 mt-0.5" />
                  <div className="flex flex-col gap-1">
                    <span className="text-xs font-bold text-rose-300">Requires Appeal Letter Support</span>
                    <p className="text-[10px] text-rose-400 leading-normal">
                      Condition severity meets criteria for insurance pre-authorization review. Check our Legal Desk to generate appeal templates.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Daily Health Tracker (FHIR Logger) */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6 shadow-md">
            <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
              <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Daily Health Tracker</h2>
                <p className="text-zinc-500 text-xs mt-0.5">Secure FHIR v4 Observation JSON logger</p>
              </div>
            </div>

            {/* Tracker Type Selector Tabs */}
            <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
              {[
                { id: "steps", label: "Steps", icon: Footprints },
                { id: "sleep", label: "Sleep", icon: Moon },
                { id: "nutrition", label: "Nutrition", icon: Flame }
              ].map((tab) => {
                const IconComponent = tab.icon;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => {
                      setLogType(tab.id);
                      setFhirLogToast(null);
                    }}
                    className={`flex-1 py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-all ${
                      logType === tab.id 
                        ? "bg-slate-850 text-emerald-400 border border-slate-700/50 shadow-sm" 
                        : "text-zinc-500 hover:text-white"
                    }`}
                  >
                    <IconComponent className="w-3.5 h-3.5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tracker Submission Form */}
            <form onSubmit={handleLogFhirObservation} className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-zinc-400 font-medium">
                  {logType === "steps" 
                    ? "Enter Daily Steps Count" 
                    : logType === "sleep" 
                    ? "Enter Sleep Duration (Hours)" 
                    : "Enter Calories Consumed (kcal)"
                  }
                </label>
                <input
                  type="number"
                  required
                  value={logValue}
                  onChange={(e) => setLogValue(e.target.value)}
                  placeholder={logType === "steps" ? "8000" : logType === "sleep" ? "7.5" : "2100"}
                  className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-emerald-500 transition-colors"
                />
              </div>

              <button
                type="submit"
                disabled={isLogging}
                className="w-full py-2.5 bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-900/50 text-slate-950 font-bold rounded-xl text-xs flex items-center justify-center gap-1.5 transition-colors"
              >
                {isLogging ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Logging FHIR Observation...</span>
                  </>
                ) : (
                  <span>Log to FHIR Server</span>
                )}
              </button>
            </form>

            {/* Logging Feedback Message (FHIR Toast Alert) */}
            {fhirLogToast && (
              <div className="bg-emerald-950/40 border border-emerald-500/30 p-3.5 rounded-xl flex items-start gap-2.5 animate-in fade-in duration-200">
                <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div className="flex flex-col gap-0.5 text-xs text-emerald-300">
                  <span className="font-bold">FHIR Observation Created</span>
                  <span className="text-[10px] font-mono text-emerald-400/80">ID: {fhirLogToast.id}</span>
                  <span className="text-[9px] text-zinc-500 font-mono">Timestamp: {fhirLogToast.timestamp}</span>
                </div>
              </div>
            )}
          </div>

        </div>

      </div>

    </div>
  );
}
