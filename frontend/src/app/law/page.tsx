"use client";

import React, { useState } from "react";
import { Scale, FileSignature, ShieldCheck, AlertTriangle, ArrowRight, Download } from "lucide-react";
import Navbar from "../../components/Navbar";

export default function LawPage() {
  // Appeal letter state
  const [denialLetter, setDenialLetter] = useState("");
  const [patientName, setPatientName] = useState("");
  const [policyId, setPolicyId] = useState("");
  const [isUrgent, setIsUrgent] = useState(false);
  const [isGeneratingAppeal, setIsGeneratingAppeal] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [appealResult, setAppealResult] = useState<any>(null);

  // Contract analysis state
  const [contractText, setContractText] = useState("");
  const [isAnalyzingContract, setIsAnalyzingContract] = useState(false);
  const [contractResult, setContractResult] = useState<any>(null);

  // HIPAA state
  const [hipaaName, setHipaaName] = useState("");
  const [hipaaDob, setHipaaDob] = useState("");
  const [hipaaProvider, setHipaaProvider] = useState("");
  const [isGeneratingHipaa, setIsGeneratingHipaa] = useState(false);
  const [hipaaResult, setHipaaResult] = useState<any>(null);

  // Handlers
  const handleGenerateAppeal = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!denialLetter || !patientName || !policyId) return;
    setIsGeneratingAppeal(true);
    setAppealResult(null);

    const formData = new FormData();
    formData.append("denial_letter", denialLetter);
    formData.append("patient_name", patientName);
    formData.append("policy_id", policyId);
    if (isUrgent) {
      formData.append("is_urgent", "true");
    }

    try {
      const res = await fetch("/api/v1/law/appeal", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setAppealResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to generate appeal letter.");
    } finally {
      setIsGeneratingAppeal(false);
    }
  };

  const handleDownloadLegalPdf = async () => {
    if (!denialLetter || !patientName || !policyId) return;
    setIsDownloadingPdf(true);

    const formData = new FormData();
    formData.append("denial_letter", denialLetter);
    formData.append("patient_name", patientName);
    formData.append("policy_id", policyId);
    if (isUrgent) {
      formData.append("is_urgent", "true");
    }

    try {
      const res = await fetch("/api/v1/law/appeal/pdf", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`PDF generation failed with status ${res.status}`);
      }

      const blob = await res.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `Legal_Appeal_${policyId || "Document"}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error(err);
      alert("Failed to download formal legal PDF.");
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleAnalyzeContract = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!contractText) return;
    setIsAnalyzingContract(true);
    setContractResult(null);

    const formData = new FormData();
    formData.append("contract_text", contractText);

    try {
      const res = await fetch("/api/v1/law/redline", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setContractResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze contract.");
    } finally {
      setIsAnalyzingContract(false);
    }
  };

  const handleGenerateHipaa = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hipaaName || !hipaaDob || !hipaaProvider) return;
    setIsGeneratingHipaa(true);
    setHipaaResult(null);

    const formData = new FormData();
    formData.append("patient_name", hipaaName);
    formData.append("dob", hipaaDob);
    formData.append("provider_name", hipaaProvider);

    try {
      const res = await fetch("/api/v1/law/hipaa-request", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setHipaaResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to generate HIPAA request.");
    } finally {
      setIsGeneratingHipaa(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-10 flex flex-col gap-10 relative pb-24">
      
      {/* Title */}
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-3xl font-extrabold text-white tracking-tight">Legal Advocacy Desk</h2>
        <p className="text-zinc-400 text-sm">Regulatory claim appeals (ACA § 2719), No Surprises Act contract redlining, and statutory HIPAA (45 CFR § 164.508) record request generators.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Col 1: Insurance Appeal & HIPAA Release */}
        <div className="flex flex-col gap-8">
          
          {/* Insurance Denial Appeals */}
          <div className="glass-panel rounded-2xl p-6 flex flex-col gap-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-white/5 pb-4">
              ⚖️ Insurance Denial Appeal Builder (ACA Section 2719)
            </h3>

            <form onSubmit={handleGenerateAppeal} className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-400">Patient Name</label>
                  <input
                    type="text"
                    required
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="e.g. John Doe"
                    className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-amber-500"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-400">Policy / Claim ID</label>
                  <input
                    type="text"
                    required
                    value={policyId}
                    onChange={(e) => setPolicyId(e.target.value)}
                    placeholder="e.g. POL-999-MED"
                    className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-amber-500"
                  />
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-zinc-400">Paste Rejection Letter Content</label>
                <textarea
                  required
                  value={denialLetter}
                  onChange={(e) => setDenialLetter(e.target.value)}
                  placeholder="Paste the denial notice text from your insurance company..."
                  rows={4}
                  className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-amber-500"
                />
              </div>

              <div className="flex items-center gap-2 py-1">
                <input
                  type="checkbox"
                  id="urgentCheck"
                  checked={isUrgent}
                  onChange={(e) => setIsUrgent(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500 cursor-pointer"
                />
                <label htmlFor="urgentCheck" className="text-xs text-amber-300 font-semibold cursor-pointer select-none">
                  Urgent Care Claim (Expedited 72-Hour Review Request per 29 CFR § 2560.503-1)
                </label>
              </div>

              <button
                type="submit"
                disabled={isGeneratingAppeal}
                className="w-full py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-amber-800 text-white font-bold rounded-xl text-xs transition-colors shadow-md"
              >
                {isGeneratingAppeal ? "Drafting Legal Appeal..." : "Generate Appeal (ACA Section 2719 / 45 CFR § 147.136)"}
              </button>
            </form>

            {appealResult && (
              <div className="bg-white/[0.01] border border-white/5 rounded-xl p-4 flex flex-col gap-3 animate-in fade-in duration-300">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-400 font-bold">REASON FOR DENIAL:</span>
                  <span className="text-amber-400 font-semibold">{appealResult.denial_reason}</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-400 font-bold">APPLICABLE STATUTE:</span>
                  <span className="text-emerald-400 font-mono text-[10px]">{appealResult.applicable_statute}</span>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] text-zinc-500 font-bold uppercase">FORMAL APPEAL LETTER DRAFT</span>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => {
                          navigator.clipboard.writeText(appealResult.appeal_letter);
                          alert("Appeal letter copied to clipboard!");
                        }}
                        className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] font-bold rounded-lg transition-colors"
                      >
                        Copy Legal Text
                      </button>
                      <button 
                        onClick={handleDownloadLegalPdf}
                        disabled={isDownloadingPdf}
                        className="px-2.5 py-1 bg-amber-600/20 hover:bg-amber-600/40 text-amber-300 border border-amber-500/30 text-[10px] font-bold rounded-lg transition-colors flex items-center gap-1 disabled:opacity-60"
                      >
                        <Download className="w-3 h-3" /> {isDownloadingPdf ? "Preparing PDF..." : "Download PDF"}
                      </button>
                      <button 
                        onClick={() => {
                          const element = document.createElement("a");
                          const file = new Blob([appealResult.appeal_letter], {type: 'text/plain'});
                          element.href = URL.createObjectURL(file);
                          element.download = "ACA_2719_Insurance_Appeal.txt";
                          document.body.appendChild(element);
                          element.click();
                        }}
                        className="px-2.5 py-1 bg-amber-600/20 hover:bg-amber-600/40 text-amber-300 border border-amber-500/30 text-[10px] font-bold rounded-lg transition-colors flex items-center gap-1"
                      >
                        <Download className="w-3 h-3" /> Export Document
                      </button>
                    </div>
                  </div>
                  <pre className="text-xs text-zinc-300 font-sans whitespace-pre-wrap bg-black/60 border border-white/10 rounded-lg p-3 max-h-[300px] overflow-y-auto leading-relaxed">
                    {appealResult.appeal_letter}
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* HIPAA release request */}
          <div className="glass-panel rounded-2xl p-6 flex flex-col gap-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-white/5 pb-4">
              🛡️ Statutory HIPAA Medical Release (45 CFR § 164.508)
            </h3>

            <form onSubmit={handleGenerateHipaa} className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-400">Full Legal Name</label>
                  <input
                    type="text"
                    required
                    value={hipaaName}
                    onChange={(e) => setHipaaName(e.target.value)}
                    placeholder="John Doe"
                    className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-amber-500"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-zinc-400">Date of Birth</label>
                  <input
                    type="text"
                    required
                    value={hipaaDob}
                    onChange={(e) => setHipaaDob(e.target.value)}
                    placeholder="MM/DD/YYYY"
                    className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-amber-500"
                  />
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-zinc-400">Hospital / Provider Name</label>
                <input
                  type="text"
                  required
                  value={hipaaProvider}
                  onChange={(e) => setHipaaProvider(e.target.value)}
                  placeholder="Metro General Hospital"
                  className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-amber-500"
                />
              </div>

              <button
                type="submit"
                disabled={isGeneratingHipaa}
                className="w-full py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-amber-800 text-white font-bold rounded-xl text-xs transition-colors shadow-md"
              >
                {isGeneratingHipaa ? "Generating HIPAA Authorization..." : "Generate Medical Records Release Form"}
              </button>
            </form>

            {hipaaResult && (
              <div className="bg-white/[0.01] border border-white/5 rounded-xl p-4 flex flex-col gap-3 animate-in fade-in duration-300">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-emerald-400 font-bold">✓ Statutory HIPAA Release Generated (45 CFR § 164.508)</span>
                  <button 
                    onClick={() => {
                      const element = document.createElement("a");
                      const file = new Blob([hipaaResult.hipaa_letter], {type: 'text/plain'});
                      element.href = URL.createObjectURL(file);
                      element.download = `HIPAA_Release_${hipaaName.replace(/\s+/g, '_')}.txt`;
                      document.body.appendChild(element);
                      element.click();
                    }}
                    className="text-emerald-300 hover:text-white bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 rounded-lg flex items-center gap-1 font-bold text-xs"
                  >
                    <Download className="w-3.5 h-3.5" /> Instant Download
                  </button>
                </div>
                <pre className="text-xs text-zinc-300 font-sans whitespace-pre-wrap bg-black/60 border border-white/10 rounded-lg p-3 max-h-[300px] overflow-y-auto leading-relaxed">
                  {hipaaResult.hipaa_letter}
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Col 2: Side-by-side Contract Redliner */}
        <div className="glass-panel rounded-2xl p-6 flex flex-col gap-6">
          <div className="flex justify-between items-center border-b border-white/5 pb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              📝 Hospital Contract Redliner & Risk Matrix
            </h3>
            {contractResult && (
              <button 
                type="button"
                onClick={() => {
                  let docContent = `REDLINED CONTRACT AUDIT REPORT\nGenerated via Legal Mediverse\nOverall Risk Score: ${contractResult.overall_risk_score}%\n\n`;
                  contractResult.predatory_clauses?.forEach((item: any, i: number) => {
                    docContent += `[FLAGGED CLAUSE #${i+1}] Risk Category: ${item.risk_category} | Severity: ${item.severity || 'High Risk'}\nORIGINAL: "${item.original_text}"\nEXPLANATION: ${item.explanation}\nSUGGESTED REVISION: "${item.suggested_revision}"\n\n`;
                  });
                  const element = document.createElement("a");
                  const file = new Blob([docContent], {type: 'text/plain'});
                  element.href = URL.createObjectURL(file);
                  element.download = "Redlined_Contract_Audit_Report.txt";
                  document.body.appendChild(element);
                  element.click();
                }}
                className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:text-white rounded-lg text-xs font-bold flex items-center gap-1"
              >
                <Download className="w-3.5 h-3.5" /> Download Full Redlined Document
              </button>
            )}
          </div>

          <form onSubmit={handleAnalyzeContract} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-zinc-400 font-medium">Paste Medical/Admission Contract Clauses</label>
              <textarea
                required
                value={contractText}
                onChange={(e) => setContractText(e.target.value)}
                placeholder="Paste liability waivers, billing sections, or arbitration terms..."
                rows={6}
                className="bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-amber-500"
              />
            </div>

            <button
              type="submit"
              disabled={isAnalyzingContract}
              className="w-full py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-amber-800 text-white font-bold rounded-xl text-xs transition-colors shadow-md"
            >
              {isAnalyzingContract ? "Redlining Clauses..." : "Analyze Contract & Redline Risk Matrix"}
            </button>
          </form>

          {/* Analysis Results side-by-side */}
          {contractResult && (
            <div className="flex flex-col gap-6 border-t border-white/5 pt-6 animate-in fade-in duration-300">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400 font-semibold">Overall Contract Risk Score</span>
                <span className={`text-[10px] uppercase font-bold tracking-widest px-3 py-1 rounded-full ${
                  contractResult.overall_risk_score > 60 
                    ? "bg-rose-500/10 text-rose-400 border border-rose-500/30" 
                    : "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                }`}>
                  {contractResult.overall_risk_score}% Risk
                </span>
              </div>

              {/* Loop clauses with severity badging */}
              <div className="flex flex-col gap-4">
                {contractResult.predatory_clauses?.map((item: any, idx: number) => {
                  const severity = item.severity || (item.risk_category === "billing_arbitration" ? "High Risk" : "Medium Risk");
                  return (
                    <div key={idx} className="grid grid-cols-1 md:grid-cols-2 gap-4 border border-white/10 rounded-xl overflow-hidden bg-black/30">
                      
                      {/* Predatory Side */}
                      <div className="p-4 border-b md:border-b-0 md:border-r border-white/10 flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[9px] uppercase font-extrabold tracking-widest text-rose-400 flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5" /> Predatory Clause
                          </span>
                          <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${
                            severity === "High Risk"
                              ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                              : severity === "Medium Risk"
                              ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                              : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          }`}>
                            {severity === "High Risk" ? "🔴 High Risk" : severity === "Medium Risk" ? "🟡 Medium Risk" : "🟢 Low Risk"}
                          </span>
                        </div>
                        <p className="text-xs text-zinc-400 italic mt-1">"{item.original_text}"</p>
                        <div className="mt-2 text-[10px] text-zinc-500">
                          <span className="font-bold text-white block">Explanation:</span>
                          {item.explanation}
                        </div>
                      </div>

                      {/* Revised Side */}
                      <div className="p-4 bg-emerald-950/5 flex flex-col gap-2">
                        <span className="text-[9px] uppercase font-extrabold tracking-widest text-emerald-400 flex items-center gap-1">
                          <ShieldCheck className="w-3.5 h-3.5" /> Suggested Revision
                        </span>
                        <p className="text-xs text-emerald-300 leading-relaxed font-medium">"{item.suggested_revision}"</p>
                        <button 
                          type="button"
                          onClick={() => {
                            navigator.clipboard.writeText(item.suggested_revision);
                            alert("Suggested revision copied to clipboard!");
                          }}
                          className="mt-auto flex items-center justify-end text-[10px] text-emerald-400 hover:text-emerald-300 font-semibold gap-1"
                        >
                          Copy Revision <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>

                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sticky Legal Disclaimer Footer */}
      <div className="fixed bottom-0 left-0 right-0 z-40 bg-slate-950/90 backdrop-blur-md border-t border-slate-800 py-3 px-6 text-center shadow-lg">
        <p className="text-xs text-slate-400 font-medium max-w-4xl mx-auto">
          ⚖️ <strong className="text-slate-200">Legal Mediverse</strong> is an AI-powered legal advocacy tool that generates statutory templates (ACA § 2719, HIPAA 45 CFR § 164.508). It does not provide formal attorney-client representation.
        </p>
      </div>

    </div>
  );
}
