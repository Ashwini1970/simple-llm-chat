# Chatbot

A small web chat that sends each turn to an OpenAI-compatible LLM and shows the reply.

## Run

1. Copy `.env.example` to `.env` and set `LLM_API_KEY`.
2. Start the server:

```
python server.py
```

3. Open http://127.0.0.1:8765

`LLM_BASE_URL` can point at any service that accepts `POST /v1/chat/completions`.
