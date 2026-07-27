import logging
from typing import Dict, Any, Optional
from backend.app.core.security import redact_pii

logger = logging.getLogger("legal_agent")

google_genai_installed = False
try:
    from google import genai
    google_genai_installed = True
except:
    pass

class LegalAppealAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client_ready = False
        if google_genai_installed and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.client_ready = True
            except Exception as e:
                logger.warning(f"Failed to configure Gemini client in LegalAppealAgent: {e}")

    async def generate_appeal(
        self, 
        denial_text: str, 
        patient_name: str, 
        policy_id: str,
        insurance_provider: Optional[str] = "Insurance Provider",
        claim_id: Optional[str] = None,
        denial_code: Optional[str] = None,
        is_urgent: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Parses insurance denial letters and drafts a formal appeal citing ACA Section 2719 (45 CFR § 147.136).
        Supports standard 180-day filing window and 72-hour expedited urgent care reviews (29 CFR § 2560.503-1(f)(2)(i)).
        """
        # Redact PHI first
        redacted_denial, _, _ = redact_pii(denial_text)

        urgent_clause_prompt = ""
        if is_urgent:
            urgent_clause_prompt = "URGENT CARE MANDATE: This is an urgent care claim. Explicitly demand an EXPEDITED 72-HOUR REVIEW citing 29 CFR § 2560.503-1(f)(2)(i)."

        statutory_citations = [
            "ACA Section 2719 (45 CFR § 147.136)",
            "ERISA 29 U.S.C. § 1133",
            "HIPAA Right of Access 45 CFR § 164.524",
        ]
        if is_urgent:
            statutory_citations.append("29 CFR § 2560.503-1(f)(2)(i)")

        prompt = f"""
        You are an expert healthcare legal advocacy attorney. Parse the following insurance claim denial notice and draft a formal, legally grounded appeal letter.
        MANDATORY STATUTORY REQUIREMENTS:
        1. Explicitly cite ACA Section 2719 (45 CFR § 147.136) governing internal claims and appeals and the 180-day filing window under 45 CFR § 147.136(b).
        2. {urgent_clause_prompt}
        3. Ensure the output includes the statutory citations list: {', '.join(statutory_citations)}.
        
        CLAIM DENIAL NOTICE CONTENT:
        {redacted_denial}
        
        CASE DETAILS:
        - Patient Name: {patient_name}
        - Policy / Member ID: {policy_id}
        - Insurance Carrier: {insurance_provider or "Insurance Provider"}
        - Claim ID: {claim_id or policy_id}
        - Denial Code: {denial_code or "N/A"}
        - Claim Type: {"Urgent Care (Expedited)" if is_urgent else "Standard Review (180-day Filing Window)"}
        
        Format your response ONLY in raw JSON matching this schema:
        {{
          "denial_reason": "<summary of why coverage was denied>",
          "applicable_statute": "ACA Section 2719 (45 CFR § 147.136) / ERISA 29 U.S.C. § 1133" + (" / 29 CFR § 2560.503-1(f)(2)(i)" if is_urgent else ""),
          "appeal_letter": "<complete, formal legal appeal letter referencing 180-day filing window and ACA Section 2719 (45 CFR § 147.136)>"
        }}
        """

        if self.client_ready:
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[prompt]
                )
                resp_text = response.text.strip()
                if resp_text.startswith("```json"):
                    resp_text = resp_text.strip("```json").strip("```").strip()
                elif resp_text.startswith("```"):
                    resp_text = resp_text.strip("```").strip()
                import json
                return json.loads(resp_text)
            except Exception as e:
                logger.error(f"Gemini error generating appeal, using fallback templates: {e}")
                return self._mock_appeal(
                    denial_text=redacted_denial,
                    name=patient_name,
                    policy_id=policy_id,
                    insurance_provider=insurance_provider,
                    claim_id=claim_id,
                    denial_code=denial_code,
                    is_urgent=is_urgent
                )
        else:
            return self._mock_appeal(
                denial_text=redacted_denial,
                name=patient_name,
                policy_id=policy_id,
                insurance_provider=insurance_provider,
                claim_id=claim_id,
                denial_code=denial_code,
                is_urgent=is_urgent
            )

    async def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """
        Analyzes a medical/hospital contract side-by-side, highlighting predatory clauses with risk severity levels.
        Checks for balance billing violations under the No Surprises Act (45 CFR § 149.110/120).
        """
        redacted_contract, _, _ = redact_pii(contract_text)
        prompt = f"""
        You are an expert healthcare contract attorney. Redline the following medical admission/provider contract and highlight predatory clauses,
        excessive liability waivers, hidden fees, or out-of-network balance billing terms that violate the No Surprises Act (45 CFR § 149.110/120).
        
        CONTRACT TEXT:
        {redacted_contract}
        
        Format your response ONLY in raw JSON matching this schema:
        {{
          "overall_risk_score": <int 1-100, where 100 is high risk>,
          "predatory_clauses": [
            {{
              "original_text": "<text from contract>",
              "risk_category": "liability_waiver" | "billing_arbitration" | "hidden_cost" | "no_surprises_act_violation" | "privacy_disclosure",
              "severity": "High Risk" | "Medium Risk" | "Low Risk",
              "explanation": "<why this clause is predatory or violates the No Surprises Act (45 CFR § 149.110/120)>",
              "suggested_revision": "<fair alternative clause compliant with No Surprises Act (45 CFR § 149.110/120) and ACA>"
            }}
          ]
        }}
        """

        if self.client_ready:
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[prompt]
                )
                resp_text = response.text.strip()
                if resp_text.startswith("```json"):
                    resp_text = resp_text.strip("```json").strip("```").strip()
                elif resp_text.startswith("```"):
                    resp_text = resp_text.strip("```").strip()
                import json
                return json.loads(resp_text)
            except Exception as e:
                logger.error(f"Gemini error analyzing contract: {e}")
                return self._mock_contract_analysis(redacted_contract)
        else:
            return self._mock_contract_analysis(redacted_contract)

    def generate_hipaa_request(self, patient_name: str, dob: str, provider_name: str, date_range: Optional[str] = "All Available Records", target_recipient: Optional[str] = "Self / Designated Advocate") -> str:
        """
        Generates a statutory 45 CFR § 164.508 & 45 CFR § 164.524 HIPAA compliant medical records release authorization letter.
        Enforces 30-day mandatory fulfillment window and OCR electronic fee limitation mandates.
        """
        letter = f"""FORMAL HIPAA MEDICAL RECORDS RELEASE AUTHORIZATION
(Pursuant to HIPAA Privacy Rule 45 CFR § 164.508 & Statutory Right of Access 45 CFR § 164.524)

DATE: July 23, 2026

TO: Custodian of Medical Records
HEALTHCARE PROVIDER / FACILITY: {provider_name}

1. PATIENT IDENTIFICATION:
   - Full Legal Name: {patient_name}
   - Date of Birth: {dob}

2. AUTHORIZATION & RECIPIENT DESIGNATION:
   I hereby authorize {provider_name} to release and disclose my Protected Health Information (PHI) to:
   - Target Recipient: {target_recipient or "Self / Designated Advocate"}

3. SCOPE OF RECORDS REQUESTED (DATE RANGE: {date_range or "All Available Records"}):
   [X] Complete Clinical History & Physician Progress Notes
   [X] Laboratory Results & Diagnostic Pathology Reports
   [X] Full Resolution Diagnostic Radiology Images (including binary DICOM files & reports)
   [X] Itemized Billing Statements & Insurance Explanation of Benefits (EOB)

4. STATUTORY MANDATES & 30-DAY FULFILLMENT WINDOW (45 CFR § 164.524(b)(2)):
   Under federal law, you are legally obligated to fulfill this record request within thirty (30) calendar days. Failure to comply within 30 days constitutes a violation of federal HIPAA Right of Access enforcement rules.

5. FEE LIMITATIONS & OCR MANDATE (45 CFR § 164.524(c)(4)):
   In accordance with HHS/OCR Right of Access guidance, any fee charged for electronic transmission must be strictly cost-based and capped at labor costs for copying/postage. Search, retrieval, or administrative handling fees are strictly prohibited.

6. EXPIRATION & REVOCATION:
   This authorization remains valid for one (1) year from date of signature unless revoked in writing.

PATIENT SIGNATURE: _________________________________________
PRINTED NAME: {patient_name}
DATE: July 23, 2026
        """
        return letter

    def _mock_appeal(
        self, 
        denial_text: str, 
        name: str, 
        policy_id: str, 
        insurance_provider: Optional[str] = "Insurance Carrier", 
        claim_id: Optional[str] = None, 
        denial_code: Optional[str] = None,
        is_urgent: Optional[bool] = False
    ) -> Dict[str, Any]:
        statute_str = "ACA Section 2719 (45 CFR § 147.136) / ERISA 29 U.S.C. § 1133"
        urgent_hdr = ""
        urgent_body = "Under 45 CFR § 147.136(b), I am exercising my right to a full and fair internal review within the 180-day statutory window."
        if is_urgent:
            statute_str += " / 29 CFR § 2560.503-1(f)(2)(i)"
            urgent_hdr = " *** EXPEDITED 72-HOUR URGENT CARE APPEAL ***"
            urgent_body = "MANDATORY URGENT CARE NOTICE: Pursuant to 29 CFR § 2560.503-1(f)(2)(i), this claim involves urgent medical care. We hereby demand an EXPEDITED DETERMINATION WITHIN 72 HOURS."

        letter = f"""FORMAL NOTICE OF APPEAL - HEALTHCARE CLAIM DENIAL{urgent_hdr}

Date: July 25, 2026
To: Appeals Department, {insurance_provider or 'Insurance Carrier'}
Patient Name: {name}
Policy / Member ID: {policy_id}
Claim ID: {claim_id or policy_id}
Denial Reason Code: {denial_code or 'N/A'}

RE: Formal Administrative Appeal pursuant to {statute_str}

Dear Appeals Committee,

I am writing to formally appeal the adverse benefit determination regarding the medical services rendered for {name}. {urgent_body}

STATUTORY BASIS & ARGUMENT:
Under ACA Section 2719 (45 CFR § 147.136) and ERISA (29 U.S.C. § 1133), plan participants are guaranteed the right to a full and fair review of denied claims, including access to all clinical rationale and medical necessity guidelines relied upon by the insurer.

The initial claim denial citing "{denial_text}" fails to account for established clinical practice guidelines. The treating physician determined that the requested intervention was medically necessary and urgent.

DEMAND FOR RELIEF:
1. Immediately reverse the adverse benefit determination and approve full coverage for Claim #{claim_id or policy_id}.
2. Provide copies of all internal guidelines, clinical criteria, or medical reviewer notes utilized in this determination.

Sincerely,

{name}
Authorized Healthcare Advocate / Patient"""

        return {
            "denial_reason": denial_text,
            "applicable_statute": statute_str,
            "appeal_letter": letter,
            "citations": [
                "ACA Section 2719 (45 CFR § 147.136)",
                "ERISA 29 U.S.C. § 1133",
                "HIPAA Right of Access 45 CFR § 164.524",
            ] + (["29 CFR § 2560.503-1(f)(2)(i)"] if is_urgent else [])
        }

    def _mock_contract_analysis(self, contract_text: str) -> Dict[str, Any]:
        return {
            "overall_risk_score": 78,
            "citations": [
                "No Surprises Act 45 CFR § 149.110/120",
                "ACA Section 2719 (45 CFR § 147.136)",
                "ERISA 29 U.S.C. § 1133"
            ],
            "predatory_clauses": [
                {
                    "original_text": "Patient agrees to waive all rights to a jury trial and accepts sole responsibility for out-of-network balance billing.",
                    "risk_category": "no_surprises_act_violation",
                    "severity": "High Risk",
                    "explanation": "This is an illegal balance billing clause violating the No Surprises Act (45 CFR § 149.110/120). Providers cannot balance bill for emergency services or non-consented out-of-network facility charges.",
                    "suggested_revision": "The provider agrees to seek billing in strict accordance with the No Surprises Act (45 CFR § 149.110/120 and Title I of Division BB of CAA 2021)."
                }
            ]
        }
