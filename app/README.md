# REMLight Chat Client

A modern React chat client for testing REMLight agents with SSE streaming, tool call visualization, and session management.

## Features

- Gray-scale modern theme with drop shadows
- SSE streaming with real-time tool call cards (expandable)
- Agent/model selection in chat toolbar
- Session history sidebar with search
- Feedback buttons (thumbs up/down)
- Add to scenario feature for evaluations
- Built-in Playwright E2E testing
- Docker setup with Postgres + nginx

## Quick Start

### Prerequisites

- Docker & Docker Compose (for Postgres database)
- Node.js 22+ (for frontend)
- Python 3.12+ with uv (for API)

### Recommended: Local Development (Best DX)

For the best development experience, run the API locally with hot reload while using Docker only for Postgres:

1. **Start Postgres** (from project root):
   ```bash
   docker-compose up postgres -d
   ```

2. **Start the API locally** (from project root):
   ```bash
   # Create/activate virtual environment
   uv venv && source .venv/bin/activate
   uv sync

   # Start API with hot reload
   uvicorn remlight.api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

3. **Start the frontend** (from app/ directory):
   ```bash
   npm install
   VITE_API_BASE_URL=http://localhost:8080/api npm run dev
   ```

4. Open http://localhost:3000

### Alternative: Full Docker Setup

The Docker Compose setup is provided for deployment or quick demos, but local development offers better hot reload and debugging:

1. Copy environment file:
   ```bash
   cp ../.env.example ../.env
   # Edit .env with your API keys
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Open http://localhost:3000

> **Note**: The dockerized API is best for deployment. For active development, run the API locally to get instant hot reload and better debugging.

## Running Tests

### E2E Tests (Playwright)

```bash
# Install Playwright browsers
npx playwright install

# Run tests
npx playwright test

# Run with UI
npx playwright test --ui

# View report
npx playwright show-report
```

## Project Structure

```
app/
├── src/
│   ├── api/              # API client functions
│   ├── components/
│   │   ├── chat/         # Chat components
│   │   ├── sidebar/      # Sidebar components
│   │   ├── layout/       # Layout components
│   │   └── ui/           # shadcn/ui components
│   ├── hooks/            # Custom React hooks
│   ├── lib/              # Utilities
│   ├── pages/            # Page components
│   └── types/            # TypeScript types
├── e2e/                  # Playwright tests
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── playwright.config.ts
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| VITE_API_BASE_URL | API endpoint | /api |

## API Endpoints

The chat client expects the following API endpoints:

- `POST /api/v1/chat/completions` - Chat completions with SSE streaming
- `GET /api/v1/agents` - List available agents
- `GET /api/v1/sessions` - List user sessions
- `GET /api/v1/sessions/:id/messages` - Get session messages
- `POST /api/v1/feedback` - Submit message feedback
- `POST /api/v1/scenarios` - Create test scenario

## Tech Stack

- React 18 + Vite + TypeScript
- shadcn/ui + Radix UI
- Tailwind CSS v4 (gray-scale theme)
- lucide-react (icons)
- react-markdown + react-syntax-highlighter
- Playwright (E2E testing)
