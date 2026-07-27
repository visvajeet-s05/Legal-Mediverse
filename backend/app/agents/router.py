import logging
import asyncio
from typing import Dict, Any, TypedDict, Annotated
from backend.app.core.config import settings
from backend.app.agents.triage_agent import TriageAgent
from backend.app.agents.legal_agent import LegalAppealAgent

logger = logging.getLogger("router_agent")

# Try to import LangGraph components
langgraph_available = False
try:
    from langgraph.graph import StateGraph, END
    langgraph_available = True
except Exception as e:
    logger.warning(f"LangGraph not installed. Falling back to custom State Machine routing: {e}")

# Define state structure
class AgentState(TypedDict):
    patient_description: str
    severity: str # mild, moderate, severe, critical
    diagnosis: str
    treatment: str
    requires_appeal: bool
    appeal_letter: str
    campaign_prefilled: Dict[str, Any]
    recovery_guide: Dict[str, Any]
    websocket_broadcast_queue: list
    differential_diagnoses: list

class RouterAgent:
    def __init__(self):
        self.triage_agent = TriageAgent(api_key=settings.GEMINI_API_KEY)
        self.legal_agent = LegalAppealAgent(api_key=settings.GEMINI_API_KEY)

    async def route_event(self, description: str, image_bytes: bytes = None) -> Dict[str, Any]:
        """
        Coordinates Multi-Agent triage, legal appeal, crowdfunding, and educational guide routing.
        """
        logger.info("Initializing multi-agent graph routing...")
        
        # Initial State
        state: AgentState = {
            "patient_description": description,
            "severity": "mild",
            "diagnosis": "",
            "treatment": "",
            "requires_appeal": False,
            "appeal_letter": "",
            "campaign_prefilled": {},
            "recovery_guide": {},
            "websocket_broadcast_queue": [],
            "differential_diagnoses": []
        }

        # Run State Machine Nodes
        state = await self._node_triage(state, image_bytes)
        
        # Cross-Domain Dispatcher Conditions
        if state["severity"] in ["severe", "critical"]:
            state["requires_appeal"] = True
            # Run parallel nodes for law, community, and edu
            legal_task = self._node_generate_legal_appeal(state)
            crowdfund_task = self._node_prefill_campaign(state)
            edu_task = self._node_generate_recovery_guide(state)
            
            # Wait for all cross-domain operations
            legal_res, crowdfund_res, edu_res = await asyncio.gather(legal_task, crowdfund_task, edu_task)
            
            state.update(legal_res)
            state.update(crowdfund_res)
            state.update(edu_res)
            
            # Broadcast the cross-domain event to WebSocket clients
            state["websocket_broadcast_queue"].append({
                "event": "CRITICAL_INJURY_DETECTED",
                "severity": state["severity"],
                "diagnosis": state["diagnosis"],
                "suggested_next_steps": {
                    "legal": "Appeal drafted citing ACA Section 2719.",
                    "community": "Crowdfunding campaign pre-filled.",
                    "education": "Patient recovery guide prepared."
                }
            })
        else:
            # For mild/moderate injuries, we still generate a recovery guide
            edu_res = await self._node_generate_recovery_guide(state)
            state.update(edu_res)

        return state

    async def _node_triage(self, state: AgentState, image_bytes: bytes = None) -> AgentState:
        logger.info("Running Node: Triage")
        triage_data = await self.triage_agent.analyze(state["patient_description"], image_bytes)
        
        state["severity"] = triage_data.get("severity", "mild")
        diagnoses = triage_data.get("differential_diagnoses", [])
        state["primary_concern"] = triage_data.get("primary_concern")
        state["icd_10_code"] = triage_data.get("icd_10_code")
        state["confidence_score"] = triage_data.get("confidence_score")
        state["citations"] = triage_data.get("citations", [])
        state["sources"] = triage_data.get("sources", [])
        if diagnoses:
            state["diagnosis"] = diagnoses[0].get("condition", "Unknown Soft Tissue Condition")
        else:
            state["diagnosis"] = "Unknown Injury"
        state["treatment"] = triage_data.get("recommended_immediate_treatment", "")
        state["differential_diagnoses"] = diagnoses
        state["risk_level"] = triage_data.get("risk_level")
        state["summary"] = triage_data.get("summary")
        state["diagnoses"] = triage_data.get("diagnoses")
        if "summary" in triage_data:
            state["recovery_guide"]["summary"] = triage_data["summary"]
        return state

    async def _node_generate_legal_appeal(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Running Node: Legal Appeal Generation")
        appeal_data = await self.legal_agent.generate_appeal(
            denial_text=f"Coverage denied for clinical procedure associated with {state['diagnosis']}.",
            patient_name="Valued Patient (Redacted)",
            policy_id="POL-999-MED"
        )
        return {
            "appeal_letter": appeal_data.get("appeal_letter", "")
        }

    async def _node_prefill_campaign(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Running Node: Crowdfund Campaign Prefill")
        # Estimate medical cost based on severity
        estimated_bill = 15000.00 if state["severity"] == "critical" else 5000.00
        
        prefilled = {
            "title": f"Medical Support for Treatment of {state['diagnosis']}",
            "description": f"This campaign is pre-filled to cover the estimated hospital and rehabilitation bills for {state['diagnosis']}. Treatment involves: {state['treatment']}.",
            "target_amount": estimated_bill,
            "bill_verification_status": "pending",
            "total_bill_amount": estimated_bill
        }
        return {
            "campaign_prefilled": prefilled
        }

    async def _node_generate_recovery_guide(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Running Node: Educational Recovery Guide")
        # Structure the recovery guide
        guide = {
            "title": f"Complete Recovery Guide for {state['diagnosis']}",
            "chapters": [
                {
                    "title": "Chapter 1: Immediate First Aid & Rest Care",
                    "content": f"To address {state['diagnosis']}: {state['treatment']}"
                },
                {
                    "title": "Chapter 2: Long-Term Rehabilitation",
                    "content": "Gradually reintroduce movement. Consult physical therapy and monitor pain levels daily."
                }
            ],
            "flashcards": [
                {"question": f"What is the first step of treatment for {state['diagnosis']}?", "answer": state["treatment"]},
                {"question": "When should you consult a doctor immediately?", "answer": "If pain increases, swelling doesn't improve, or red flags appear."}
            ]
        }
        return {
            "recovery_guide": guide
        }

router_agent = RouterAgent()
