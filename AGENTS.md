# Agent Operational Guidelines (AGENTS.md)

Welcome to the Legal Mediverse codebase. As the Lead Principal Engineer and Autonomous Systems Architect, we adhere to the following operational guidelines for development, styling, and testing.

## 1. Code Styling & Quality

- **Python Backend (FastAPI)**:
  - Adhere to PEP 8 standards.
  - Use explicit type hinting on all functions, classes, and router endpoints.
  - Implement Pydantic (v2) models for request validation and response serialization.
  - Use async-first paradigms (`async def` and `await`) for database operations, external API calls, and agent coordination.
  - Wrap database operations in transactional blocks.
- **Frontend (Next.js & TypeScript)**:
  - Use TypeScript strictly. Avoid the `any` type; define explicit interfaces and types.
  - Use Functional Components with React Hooks.
  - Keep styling consistent with Tailwind CSS utility classes.
- **Solidity (Smart Contracts)**:
  - Match Solidity version `^0.8.20`.
  - Follow the OpenZeppelin standard practices for security, ownership, and math.

## 2. Directory Layout & Rules

All new files should be created in their appropriate directories as defined in the layout:
- `backend/app/core/`: Contains server configuration, security pipelines, and database adapters.
- `backend/app/models/`: Holds database schemas and FHIR v4 Pydantic validation structures.
- `backend/app/agents/`: Holds LangGraph nodes, routing logic, and supervisors.
- `backend/app/services/`: Core external clients (Qdrant, LiveKit, Chainlink, etc.).
- `backend/app/api/v1/`: API route files grouped by service area (/health, /edu, /community, /law).
- `contracts/`: Solidity smart contracts.
- `frontend/src/`: Next.js frontend code.

## 3. Testing Command Flags

To run tests in the non-interactive environments:
- For Python tests:
  ```bash
  .venv\Scripts\pytest -v backend/tests/
  ```
- Make sure to use standard non-interactive terminal flags and avoid running blocking servers.
