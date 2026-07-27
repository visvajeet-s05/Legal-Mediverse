# Legal Mediverse

Enterprise-grade multi-agent platform for health, education, community crowdfunding, and legal services.

## Short description of this repository

## Overview

Legal Mediverse is a full-stack Web3 platform combining AI-powered medical bill verification, crowdfunding escrow smart contracts, education tools, and legal services.

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Wagmi, Viem
- **Backend**: FastAPI, SQLAlchemy, Async MySQL/SQLite
- **AI**: Google Gemini Vision OCR, Presidio PII
- **Web3**: Polygon Amoy testnet, smart contract escrow
- **Infrastructure**: Docker, Prometheus, Grafana, Redis, Qdrant

## Key Features

- Medical crowdfunding with Polygon smart contract escrow
- AI hospital bill OCR verification via Gemini Vision
- Role-based access and immutable audit ledger
- Education notes with flashcards and React Flow graphs
- Legal case automation and document redlining
- Real-time WebSocket updates and Prometheus metrics

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Environment

Copy `.env.example` to `.env.testnet` or `.env` and fill required values.

## Scripts

- `scripts/test_e2e_integration.py` - End-to-end tests
- `scripts/start_backend.py` - Backend startup helper
- `scripts/start_frontend.bat` - Frontend startup helper

## License

Proprietary - All rights reserved.