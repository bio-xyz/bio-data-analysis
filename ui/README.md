## Bio Code Interpreter UI

A lean demo frontend built with Next.js 14 (App Router), Tailwind CSS, and Radix UI to showcase the FastAPI agent endpoint.

### Requirements

- Node.js 18+
- Access to the FastAPI service (defaults to `http://localhost:8000`)

### Setup

```bash
npm install
cp .env.local.example .env.local
# adjust NEXT_PUBLIC_AGENT_API_BASE if the backend runs elsewhere
```

### Development

```bash
npm run dev
```

Open http://localhost:3000 to use the form.

### Production build

```bash
npm run build
npm start
```

### What the UI does

- Mirrors the FastAPI `TaskRequest` payload: task description (required), multiple file uploads (optional), and data files description (optional)
- Sends a multipart request to `POST /api/agent/run`
- Renders the structured `TaskResponse`, including the plan, answer summary/details, and any generated artifacts

If the backend responds with an error, the UI surfaces the message in a Radix `Callout` for fast debugging.
