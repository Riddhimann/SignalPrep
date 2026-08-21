# SignalPrep deployment: Vercel demo profile

SignalPrep deliberately separates its measured local runtime from its public demo runtime.

## Runtime profiles

| Concern | Local measured runtime | Vercel demo runtime |
|---|---|---|
| Application | React build served by FastAPI/Uvicorn | React static build + FastAPI Vercel Function |
| Generation | Ollama `qwen2.5:3b` | Hosted OpenAI-compatible model endpoint |
| Retrieval | Ollama BGE + TF-IDF hybrid | TF-IDF lexical RAG |
| State | Process memory | Process memory currently; optional Upstash Redis REST with a 24-hour TTL |
| Audio | Optional Faster-Whisper extra | Typed answers only |

The published benchmark remains evidence about the local `qwen2.5:3b` and hybrid-BGE runtime. It
must not be presented as a measurement of the hosted deployment. The application exposes exact
runtime provenance in `/api/health` and in each interview session.

## Current public deployment

The public demo at [signal-prep.vercel.app](https://signal-prep.vercel.app/) currently runs hosted
Qwen inference, lexical TF-IDF retrieval, typed answers, and in-memory sessions. Its exact effective
runtime is exposed by `/api/health` and every session response. A function cold start can reset an
unfinished interview until the optional Upstash settings below are activated.

## Why persistent storage is optional on Vercel

A Vercel Function is stateless: two consecutive requests can run in different Python processes.
When `SESSION_BACKEND=upstash_redis` is configured, SignalPrep writes the validated
`InterviewSession` and original RAG source texts to Upstash. When a new function instance receives
an answer, it reloads the session and rebuilds the small TF-IDF index. Both records expire according
to `SESSION_TTL_SECONDS`. Without those settings, it transparently uses process memory.

Do not upload real confidential resumes to the portfolio deployment. The demo has a short retention
window, but a public multi-user product would additionally require authentication, tenant isolation,
encryption policy, per-user deletion, abuse controls, and a privacy notice.

## Required Vercel environment variables

Copy the names and safe example values from `.env.vercel.example` into Vercel Project Settings.
Never commit real API keys or Redis tokens.

The recommended hosted Qwen profile uses:

```text
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen/qwen3.6-27b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_STRUCTURED_OUTPUT_MODE=json_object
RETRIEVAL_BACKEND=lexical_demo
SESSION_BACKEND=upstash_redis
```

## Deployment procedure

1. Push `SignalPrep` to a GitHub repository.
2. Create a Groq API key and keep it server-side as `LLM_API_KEY`.
3. Optionally add an Upstash Redis integration to the Vercel project for cross-instance sessions.
4. Import the GitHub repository into Vercel with `SignalPrep` as the root directory.
5. Add the variables from `.env.vercel.example` and deploy.
6. Verify `/`, `/api/health`, one complete interview, the report, runtime logs, and a redeployment.
7. If Upstash is enabled, confirm that the same session still loads after a redeployment or fresh
   function instance handles the request.

GitHub integration creates a preview deployment for branches and a production deployment from the
configured production branch. Failed builds should be inspected in Vercel deployment logs; Python
tests and the frontend build continue to run independently in GitHub Actions.
