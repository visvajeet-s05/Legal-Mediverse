"""
Qdrant Vector DB Seeding Script
================================
Seeds the Qdrant vector database with:
1. PubMed clinical references (mock embedded vectors)
2. ICD-10 diagnostic codes with descriptions

Run: python -m backend.scripts.seed_qdrant
"""

import asyncio
import hashlib
import logging
import random
import sys
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_qdrant")

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Install with: pip install qdrant-client")

# Vector dimension for text embeddings (using all-MiniLM-L6-v2 standard)
VECTOR_DIM = 384

# ─── PubMed Clinical References ──────────────────────────────────────────

PUBMED_REFERENCES: List[Dict[str, Any]] = [
    {
        "title": "Treatment Guidelines for Acute Soft Tissue Lacerations",
        "citation": "J. Trauma & Care, 2024",
        "summary": "Primary closure is indicated for clean wounds within 12-24 hours. Prophylactic antibiotics are optional but recommended for contaminated wounds or immunocompromised patients.",
        "keywords": ["cut", "laceration", "wound", "bleed"],
        "pubmed_id": "33001234",
    },
    {
        "title": "Diagnosis and Management of Ankle Sprains and Fractures",
        "citation": "Am. Academy of Ortho., 2023",
        "summary": "The Ottawa Ankle Rules should be used to determine the necessity of X-ray imaging. RICE protocol remains first line for acute ankle sprains.",
        "keywords": ["ankle", "sprain", "fracture", "foot", "swelling"],
        "pubmed_id": "35123456",
    },
    {
        "title": "Differential Diagnosis of Acute Abdominal Distress",
        "citation": "New Eng. J. Med, 2024",
        "summary": "Right lower quadrant pain is highly suspicious for acute appendicitis. Clinical diagnosis relies on Alvarado scoring with ultrasonography or CT imaging.",
        "keywords": ["abdominal", "pain", "stomach", "appendicitis", "nausea"],
        "pubmed_id": "36112233",
    },
    {
        "title": "Standard Care for First and Second Degree Thermal Burns",
        "citation": "Burn Care Journal, 2023",
        "summary": "Cool water application, debridement of broken blisters, and application of silver sulfadiazine or petroleum ointment are foundational first steps.",
        "keywords": ["burn", "fire", "heat", "skin", "blister"],
        "pubmed_id": "35998877",
    },
    {
        "title": "Chest Pain Evaluation and Acute Coronary Syndrome Guidelines",
        "citation": "AHA Circulation, 2024",
        "summary": "Immediate ECG within 10 minutes, high-sensitivity troponin testing, and aspirin 325 mg for suspected ACS using HEART score stratification.",
        "keywords": ["chest pain", "heart", "cardiac", "coronary", "ecg"],
        "pubmed_id": "34123456",
    },
    {
        "title": "Management of Acute Dyspnea and Respiratory Distress",
        "citation": "Chest Journal, 2024",
        "summary": "Assess airway, breathing, circulation. Administer supplemental oxygen to maintain SpO2 > 92%. Consider non-invasive ventilation for acute respiratory failure.",
        "keywords": ["dyspnea", "breathing", "respiratory", "shortness of breath"],
        "pubmed_id": "37234567",
    },
    {
        "title": "Stroke Recognition and Acute Ischemic Stroke Protocol",
        "citation": "Stroke Journal, 2024",
        "summary": "Use FAST (Face, Arm, Speech, Time) for rapid stroke recognition. CT scan within 20 minutes. IV alteplase within 3-4.5 hours of symptom onset.",
        "keywords": ["stroke", "facial droop", "weakness", "speech"],
        "pubmed_id": "38345678",
    },
    {
        "title": "Anaphylaxis Emergency Management Guidelines",
        "citation": "J. Allergy Clin. Immunol., 2024",
        "summary": "Epinephrine IM 0.3-0.5 mg (1:1000) anterolateral thigh is first-line treatment at first sign of anaphylaxis.",
        "keywords": ["anaphylaxis", "allergy", "hives", "swelling", "epinephrine"],
        "pubmed_id": "39456789",
    },
    {
        "title": "Seizure Emergency Management and Status Epilepticus Protocol",
        "citation": "Epilepsia Journal, 2024",
        "summary": "Benzodiazepines (lorazepam IV 0.1 mg/kg or midazolam IM 10 mg) are first-line for prolonged seizures (>5 min).",
        "keywords": ["seizure", "convulsion", "epilepsy", "status epilepticus"],
        "pubmed_id": "40567890",
    },
]

# ─── ICD-10 Diagnostic Codes ─────────────────────────────────────────────

ICD10_CODES: List[Dict[str, Any]] = [
    {"code": "I20.9", "description": "Unstable angina", "keywords": ["chest", "angina", "cardiac"]},
    {"code": "I21.4", "description": "Non-ST elevation myocardial infarction (NSTEMI)", "keywords": ["heart attack", "nstemi", "myocardial"]},
    {"code": "S93.4", "description": "Sprain and strain of ankle", "keywords": ["ankle", "sprain", "strain"]},
    {"code": "S01.0", "description": "Open wound of scalp", "keywords": ["cut", "scalp", "head", "wound"]},
    {"code": "K35.8", "description": "Acute appendicitis, other and unspecified", "keywords": ["appendicitis", "abdominal", "stomach"]},
    {"code": "T30.0", "description": "Burn of unspecified body region", "keywords": ["burn", "thermal", "fire"]},
    {"code": "M54.5", "description": "Low back pain", "keywords": ["back", "spine", "pain"]},
    {"code": "R06.0", "description": "Dyspnea", "keywords": ["dyspnea", "shortness of breath"]},
    {"code": "I63.9", "description": "Cerebral infarction, unspecified", "keywords": ["stroke", "cerebral", "infarction"]},
    {"code": "T78.2", "description": "Anaphylactic shock, unspecified", "keywords": ["anaphylaxis", "shock", "allergic"]},
    {"code": "G40.9", "description": "Epilepsy, unspecified", "keywords": ["seizure", "epilepsy", "convulsion"]},
    {"code": "S06.9", "description": "Intracranial injury, unspecified", "keywords": ["head injury", "trauma"]},
    {"code": "J45.9", "description": "Asthma, unspecified", "keywords": ["asthma", "wheezing"]},
    {"code": "N17.9", "description": "Acute kidney failure, unspecified", "keywords": ["kidney", "renal", "failure"]},
    {"code": "E10.9", "description": "Type 1 diabetes without complications", "keywords": ["diabetes", "type 1", "insulin"]},
    {"code": "E11.9", "description": "Type 2 diabetes without complications", "keywords": ["diabetes", "type 2"]},
    {"code": "I10", "description": "Essential (primary) hypertension", "keywords": ["hypertension", "blood pressure"]},
    {"code": "J15.9", "description": "Bacterial pneumonia, unspecified", "keywords": ["pneumonia", "lung", "infection"]},
    {"code": "N39.0", "description": "Urinary tract infection, site not specified", "keywords": ["uti", "urinary", "infection"]},
    {"code": "L03.9", "description": "Cellulitis, unspecified", "keywords": ["cellulitis", "skin infection"]},
]


def generate_dense_vector(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """
    Generate a deterministic dense embedding vector based on text content.
    For production, use sentence-transformers or OpenAI embeddings.
    """
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:4], "big")
    rng = random.Random(seed)
    return [rng.uniform(-0.5, 0.5) for _ in range(dim)]


async def seed_collections(qdrant_host: str = "localhost", qdrant_port: int = 6333):
    """Main seeding function."""
    if not QDRANT_AVAILABLE:
        logger.error("qdrant-client not available. Install with: pip install qdrant-client")
        return False

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")

        # ── Seed PubMed Collection ──────────────────────────────────────────
        pubmed_exists = client.collection_exists("pubmed")
        if pubmed_exists:
            logger.info("PubMed collection already exists. Recreating...")
            client.delete_collection("pubmed")

        client.create_collection(
            collection_name="pubmed",
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created PubMed collection")

        pubmed_points = []
        for idx, ref in enumerate(PUBMED_REFERENCES):
            text_for_embedding = f"{ref['title']} {ref['summary']} {' '.join(ref['keywords'])}"
            vector = generate_dense_vector(text_for_embedding)
            pubmed_points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload={
                        "title": ref["title"],
                        "citation": ref["citation"],
                        "summary": ref["summary"],
                        "keywords": ref["keywords"],
                        "pubmed_id": ref["pubmed_id"],
                    },
                )
            )

        client.upsert(collection_name="pubmed", points=pubmed_points)
        logger.info(f"Inserted {len(pubmed_points)} PubMed references")

        # ── Seed ICD-10 Collection ───────────────────────────────────────────
        icd10_exists = client.collection_exists("icd10")
        if icd10_exists:
            logger.info("ICD-10 collection already exists. Recreating...")
            client.delete_collection("icd10")

        client.create_collection(
            collection_name="icd10",
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created ICD-10 collection")

        icd10_points = []
        for idx, code in enumerate(ICD10_CODES):
            text_for_embedding = f"{code['code']} {code['description']} {' '.join(code['keywords'])}"
            vector = generate_dense_vector(text_for_embedding)
            icd10_points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload={
                        "code": code["code"],
                        "description": code["description"],
                        "keywords": code["keywords"],
                    },
                )
            )

        client.upsert(collection_name="icd10", points=icd10_points)
        logger.info(f"Inserted {len(icd10_points)} ICD-10 codes")

        logger.info("=" * 60)
        logger.info("Qdrant seeding complete!")
        logger.info(f"   - PubMed collection: {len(PUBMED_REFERENCES)} references")
        logger.info(f"   - ICD-10 collection: {len(ICD10_CODES)} codes")
        logger.info("=" * 60)

        # Test search
        test_result = client.search(
            collection_name="pubmed",
            query_vector=generate_dense_vector("chest pain heart attack"),
            limit=3,
        )
        logger.info(f"Test search 'chest pain' returned {len(test_result)} results")
        for hit in test_result:
            logger.info(f"  -> {hit.payload['title']} (score: {hit.score:.4f})")

        return True

    except Exception as e:
        logger.error(f"Failed to seed Qdrant: {e}")
        return False


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Seed Qdrant vector database with medical references")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    args = parser.parse_args()

    success = asyncio.run(seed_collections(args.host, args.port))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

