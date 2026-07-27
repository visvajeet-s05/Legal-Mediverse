# Legal Mediverse

> Enterprise-grade multi-agent platform for health, education, community crowdfunding, and legal services.

## Short description of this repository

Legal Mediverse is a full-stack Web3 platform combining AI-powered medical bill verification, Polygon-based crowdfunding escrow smart contracts, education tools, and legal services. It enables transparent medical fundraising, automated bill OCR verification, learning resources, and legal document processing.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Running Backend](#running-backend)
- [Running Frontend](#running-frontend)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

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

### Infrastructure
- **Database**: MySQL 8.0 / SQLite
- **Cache**: Redis
- **Vector DB**: Qdrant
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Docker Compose

### Blockchain
- **Network**: Polygon Amoy Testnet
- **Smart Contracts**: Solidity 0.8.20
- **Development**: Hardhat

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm
- MySQL 8.0 or SQLite
- Redis (optional, falls back to in-memory)
- Qdrant (optional, falls back to mock)
- Polygon RPC endpoint

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

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
.venv\Scripts\pytest -v backend/tests/

# Run specific test file
.venv\Scripts\pytest -v backend/tests/test_api.py

# Run with coverage
.venv\Scripts\pytest --cov=backend backend/tests/
```

### Smart Contract Tests

```bash
cd contracts

npm install
npx hardhat test
```

### E2E Integration Tests

```bash
cd scripts
python test_e2e_integration.py
```

## API Reference

### Community Endpoints

- `GET /api/v1/community/campaigns` - List all campaigns
- `POST /api/v1/community/campaigns` - Create campaign
- `GET /api/v1/community/campaigns/{id}` - Get campaign details
- `POST /api/v1/community/campaigns/{id}/donate` - Donate to campaign
- `POST /api/v1/community/campaigns/{id}/verify-bill` - Verify hospital bill
- `POST /api/v1/community/campaigns/{id}/release-milestone` - Release funds
- `POST /api/v1/community/campaigns/{id}/claim-refund` - Claim refund

### Health Endpoints

- `POST /api/v1/health/records` - Create health record
- `GET /api/v1/health/records` - List user health records
- `GET /api/v1/health/records/{id}` - Get record details

### Education Endpoints

- `POST /api/v1/edu/notes` - Create educational note
- `GET /api/v1/edu/notes` - List user notes
- `POST /api/v1/edu/notes/{id}/flashcards` - Generate flashcards

### Legal Endpoints

- `POST /api/v1/law/cases` - Create legal case
- `POST /api/v1/law/cases/{id}/analyze` - Analyze document
- `POST /api/v1/law/appeal` - Generate appeal letter

### Authentication

- `POST /api/v1/auth/register` - Register user
- `POST /api/v1/auth/login` - Login user
- `GET /api/v1/auth/me` - Get current user

## Deployment

### Docker Deployment

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Configuration

1. Set `APP_ENV=production` or `APP_ENV=testnet`
2. Configure production database
3. Set up Redis and Qdrant
4. Configure object storage (S3 or local)
5. Set up monitoring (Prometheus, Grafana)
6. Configure Sentry for error tracking
7. Set strong `JWT_SECRET_KEY`
8. Configure CORS origins
9. Set up SSL/TLS certificates

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

## Development Guidelines

### Code Style

- **Backend (Python)**:
  - PEP 8 compliance
  - Type hints on all functions and classes
  - Pydantic v2 models for validation
  - Async-first approach (`async def` and `await`)

- **Frontend (TypeScript)**:
  - Strict TypeScript mode
  - Functional components with hooks
  - Tailwind CSS utility classes
  - Explicit interfaces, avoid `any`

- **Smart Contracts (Solidity)**:
  - Solidity 0.8.20
  - OpenZeppelin standards
  - Comprehensive NatSpec comments

### Git Workflow

1. Create feature branch from `main`
2. Implement changes with clear commit messages
3. Ensure tests pass
4. Submit pull request

## Security

- PII redaction with Microsoft Presidio
- JWT-based authentication
- Role-based access control
- Immutable audit logging
- Rate limiting middleware
- Secret scanning in CI/CD
- SQL injection prevention via ORM

## Monitoring

- **Metrics**: Prometheus endpoint at `/metrics`
- **Logging**: Structured JSON logging
- **Error Tracking**: Sentry integration
- **Health Check**: `/api/health`
- **WebSocket**: Real-time event broadcasting

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