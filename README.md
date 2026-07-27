# Legal Mediverse

> Enterprise-grade multi-agent platform for health, education, community crowdfunding, and legal services.

Legal Mediverse is a full-stack Web3 platform combining AI-powered medical bill verification, Polygon-based crowdfunding escrow smart contracts, education tools, and legal services. It enables transparent medical fundraising, automated bill OCR verification, learning resources, and legal document processing.

## Features

### Community & Crowdfunding
- Create and manage medical crowdfunding campaigns
- Donations via Polygon Amoy testnet
- Smart contract escrow vault for transparent fund holding
- Gemini Vision OCR hospital bill verification
- Fraud risk scoring and anomaly detection
- Milestone-based fund release to hospital wallets
- Donor refund workflow for failed verifications
- Real-time campaign progress tracking

### Health
- FHIR v4 compliant health record storage
- Nutrition, sleep, and physical activity tracking
- DICOM viewer integration
- PII redaction with Presidio
- Health bot assistance

### Education
- Note-taking with React Flow graphs
- Flashcard generation
- Multimedia educational resources
- Online tutor integration

### Legal
- Contract review and clause highlighting
- Regulatory appeal letter generation
- HIPAA request automation
- Document redlining

### Platform
- Role-based access control
- Immutable audit ledger
- Real-time WebSocket updates
- Prometheus metrics and monitoring
- Qdrant vector DB for semantic search
- Rate limiting and security middleware

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│  Database  │
│ Next.js/TS  │     │  FastAPI    │     │ MySQL/SQLite│
└──────┬──────┘     └──────┬──────┘     └─────────────┘
       │                   │
       │              ┌────┴────┐
       │              │ Agents  │
       │              └────┬────┘
       │                   │
       │              ┌────┴────┐
       │              │ Web3    │
       │              │Escrow   │
       │              └─────────┘
       │
       ▼
Polygon Amoy Testnet
```

## Tech Stack

### Frontend
- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Web3**: Wagmi, Viem
- **UI Components**: Lucide React

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **ORM**: SQLAlchemy 2.0 (Async)
- **Validation**: Pydantic v2
- **AI/ML**: Google Gemini Vision, Presidio

## Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt

# Run server
uvicorn app.main:app --reload --port 8000
```

Backend available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend available at: `http://localhost:3000`

## Environment Setup

1. Copy `.env.example` to `.env.testnet` or `.env`
2. Configure required environment variables:

### Required Variables

```env
APP_ENV=development
DATABASE_URL=mysql+aiomysql://username:password@127.0.0.1:3306/legal_mediverse
JWT_SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
POLYGON_AMOY_RPC_URL=https://rpc-amoy.polygon.technology
ESCROW_CONTRACT_ADDRESS=0x... # Deployed contract address
```

### Optional Variables

```env
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=your-openai-key
LIVEKIT_API_KEY=your-livekit-key
S3_BUCKET=your-bucket-name
```

See `.env.example` for complete configuration options.

## Running Backend

```bash
# From project root
cd backend

# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Start server with auto-reload
uvicorn app.main:app --reload --port 8000

# Or using Python module
python -m uvicorn app.main:app --reload --port 8000
```

## Running Frontend

```bash
# From project root
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents (fraud, legal, triage)
│   │   ├── api/v1/          # API route handlers
│   │   ├── core/            # Configuration, database, security
│   │   ├── models/          # SQLAlchemy ORM models
│   │   └── services/        # External service clients
│   ├── migrations/          # Alembic DB migrations
│   ├── scripts/             # Utility scripts
│   └── tests/               # Backend tests
├── contracts/               # Solidity smart contracts
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   ├── components/      # Reusable UI components
│   │   ├── hooks/           # Custom React hooks
│   │   └── lib/             # Utilities and configs
│   └── public/              # Static assets
├── monitoring/              # Prometheus, Grafana configs
└── scripts/                 # Deployment and utility scripts
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

Proprietary - All rights reserved.

## Support

For issues and feature requests, please use the GitHub issue tracker.
