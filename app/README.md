# REMLight Chat Client

A simple React app for testing REMLight agents. Use it to chat with agents, review tool call activity, and browse session history.

## Quick Start

### Prerequisites

- Docker (for Postgres)
- Node.js 22+
- Python 3.12+ with uv (for API)

### Local Development

1. **Start Postgres** (from project root):
   ```bash
   docker-compose up postgres -d
   ```

2. **Start the API** (from project root):
   ```bash
   uv venv && source .venv/bin/activate
   uv sync
   uvicorn remlight.api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

3. **Start the frontend** (from app/ directory):
   ```bash
   npm install
   VITE_API_BASE_URL=http://localhost:8080/api npm run dev
   ```

4. Open http://localhost:3000

## Running Tests

```bash
npx playwright install
npx playwright test
```

## Project Structure

```
app/
├── src/
│   ├── api/              # API client functions
│   ├── components/
│   │   ├── chat/         # Chat view, messages, tool cards
│   │   ├── sidebar/      # Session list and search
│   │   └── ui/           # shadcn/ui components
│   ├── hooks/            # use-chat, use-sse, use-sessions
│   └── types/            # TypeScript types
├── e2e/                  # Playwright tests
└── playwright.config.ts
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| VITE_API_BASE_URL | API endpoint | /api |
