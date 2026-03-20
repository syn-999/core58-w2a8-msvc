import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:
    print(
        "Missing dependency 'torch'. Activate `venv_gpu` and install the CUDA-enabled "
        "PyTorch wheel documented in README.md before using gpu_server.py."
    )
    sys.exit(1)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    print("Missing GPU server dependencies. Activate your repo GPU venv or run `pip install -r requirements.txt` before using gpu_server.py.")
    sys.exit(1)

try:
    from gpu_generate import FastGen, GenArgs
    from tokenizer import ChatFormat
except ImportError:
    from .gpu_generate import FastGen, GenArgs
    from .tokenizer import ChatFormat

# Global config
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = THIS_DIR.parent / "models" / "gpu" / "bitnet-b1.58-2B-4T-bf16"
CKPT_DIR = os.getenv("BITNET_CKPT_DIR", str(DEFAULT_CKPT_DIR if DEFAULT_CKPT_DIR.exists() else (THIS_DIR / "checkpoints")))
DEVICE = os.getenv("BITNET_DEVICE", "cuda:0")
DECODE_BACKEND = os.getenv("BITNET_DECODE_BACKEND", "int2")
PROMPT_LENGTH = int(os.getenv("BITNET_PROMPT_LENGTH", "256"))
MAX_TOKENS = int(os.getenv("BITNET_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.getenv("BITNET_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("BITNET_TOP_P", "0.9"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "BITNET_SYSTEM_PROMPT",
    "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete.",
)

g = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global g
    print(
        f"Loading model on {DEVICE} from {CKPT_DIR} using {DECODE_BACKEND} decode "
        f"(prompt_length={PROMPT_LENGTH}, max_tokens={MAX_TOKENS})"
    )
    torch.cuda.set_device(DEVICE)
    g = FastGen.build(
        CKPT_DIR,
        GenArgs(
            gen_length=MAX_TOKENS,
            prompt_length=PROMPT_LENGTH,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
        ),
        DEVICE,
        decode_backend=DECODE_BACKEND,
    )
    g.tokenizer = ChatFormat(g.tokenizer)
    print("Model loaded and ready for inference.")
    yield

app = FastAPI(lifespan=lifespan)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>core58 GPU Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f4f8;
      --panel: #ffffff;
      --text: #17212b;
      --muted: #5a6a7c;
      --accent: #0a6cff;
      --border: #d8e0ea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
      color: var(--text);
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 12px 40px rgba(23, 33, 43, 0.08);
    }
    .hero {
      padding: 24px;
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 28px;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }
    .controls, .chat {
      padding: 20px;
    }
    .controls label, .chat label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      color: var(--text);
      background: #fbfcfe;
      margin-bottom: 14px;
    }
    textarea {
      min-height: 110px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .chat-log {
      height: 420px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fbfcfe;
      padding: 14px;
      margin-bottom: 14px;
    }
    .msg {
      padding: 12px 14px;
      border-radius: 12px;
      margin-bottom: 10px;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .msg.user {
      background: #deebff;
    }
    .msg.assistant {
      background: #eef2f7;
    }
    .msg.system {
      background: #f7f1dd;
    }
    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button.secondary {
      background: #dfe7f3;
      color: var(--text);
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
      margin-top: 8px;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .chat-log { height: 320px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel hero">
      <h1>core58 GPU Chat</h1>
      <p>Local browser chat for the Windows-native GPU runtime. This UI calls <code>/v1/chat/completions</code> on the same FastAPI server.</p>
    </section>
    <section class="grid">
      <aside class="panel controls">
        <label for="systemPrompt">System prompt</label>
        <textarea id="systemPrompt">__SYSTEM_PROMPT__</textarea>
        <div class="row">
          <div>
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" min="0" max="2" step="0.1" value="__TEMPERATURE__">
          </div>
          <div>
            <label for="topP">Top-p</label>
            <input id="topP" type="number" min="0" max="1" step="0.05" value="__TOP_P__">
          </div>
        </div>
        <label for="maxTokens">Max new tokens</label>
        <input id="maxTokens" type="number" min="1" max="__MAX_TOKENS__" step="1" value="256">
        <div class="status">Backend: __BACKEND__ | Prompt limit: __PROMPT_LENGTH__ tokens | Max decode: __MAX_TOKENS__ tokens</div>
        <div class="status"><a href="/docs">Open API docs</a></div>
      </aside>
      <section class="panel chat">
        <div id="chatLog" class="chat-log"></div>
        <label for="userInput">Message</label>
        <textarea id="userInput" placeholder="Ask something..."></textarea>
        <div class="actions">
          <button id="sendBtn">Send</button>
          <button id="resetBtn" class="secondary">Reset</button>
        </div>
        <div id="status" class="status"></div>
      </section>
    </section>
  </main>
  <script>
    const chatLog = document.getElementById('chatLog');
    const userInput = document.getElementById('userInput');
    const systemPrompt = document.getElementById('systemPrompt');
    const temperature = document.getElementById('temperature');
    const topP = document.getElementById('topP');
    const maxTokens = document.getElementById('maxTokens');
    const statusEl = document.getElementById('status');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');

    let messages = [];

    function render() {
      chatLog.innerHTML = '';
      const systemValue = systemPrompt.value.trim();
      if (systemValue) {
        appendBubble('system', systemValue);
      }
      for (const msg of messages) {
        appendBubble(msg.role, msg.content);
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function appendBubble(role, content) {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      const label = document.createElement('strong');
      label.textContent = `${role[0].toUpperCase()}${role.slice(1)}:`;
      const body = document.createElement('div');
      body.textContent = content;
      node.appendChild(label);
      node.appendChild(document.createElement('br'));
      node.appendChild(body);
      chatLog.appendChild(node);
    }

    function buildPayload() {
      const dialog = [];
      const systemValue = systemPrompt.value.trim();
      if (systemValue) {
        dialog.push({ role: 'system', content: systemValue });
      }
      dialog.push(...messages);
      return {
        model: 'core58-gpu',
        messages: dialog,
        temperature: Number(temperature.value),
        top_p: Number(topP.value),
        max_tokens: Number(maxTokens.value),
        stream: false,
      };
    }

    async function send() {
      const content = userInput.value.trim();
      if (!content) return;
      messages.push({ role: 'user', content });
      userInput.value = '';
      render();
      sendBtn.disabled = true;
      statusEl.textContent = 'Generating...';
      try {
        const resp = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildPayload()),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        const answer = data.choices?.[0]?.message?.content?.trim() || '';
        messages.push({ role: 'assistant', content: answer });
        statusEl.textContent = `Done. Prompt: ${data.usage.prompt_tokens} | Completion: ${data.usage.completion_tokens}`;
        render();
      } catch (err) {
        statusEl.textContent = err.message;
      } finally {
        sendBtn.disabled = false;
      }
    }

    sendBtn.addEventListener('click', send);
    userInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        send();
      }
    });
    resetBtn.addEventListener('click', () => {
      messages = [];
      statusEl.textContent = 'Conversation reset.';
      render();
    });
    render();
  </script>
</body>
</html>
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "core58-gpu"
    messages: List[ChatMessage]
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

def encode_completion_dialog(
    tokenizer: ChatFormat,
    message_token_groups: List[List[int]],
    assistant_header: List[int],
) -> List[int]:
    bos_id = tokenizer.tokenizer.special_tokens["<|begin_of_text|>"]
    tokens = [bos_id]
    for message_tokens in message_token_groups:
        tokens.extend(message_tokens)
    tokens.extend(assistant_header)
    return tokens

@app.get("/", response_class=HTMLResponse)
async def index():
    page = (
        INDEX_HTML.replace("__SYSTEM_PROMPT__", DEFAULT_SYSTEM_PROMPT)
        .replace("__TEMPERATURE__", str(DEFAULT_TEMPERATURE))
        .replace("__TOP_P__", str(DEFAULT_TOP_P))
        .replace("__MAX_TOKENS__", str(MAX_TOKENS))
        .replace("__PROMPT_LENGTH__", str(PROMPT_LENGTH))
        .replace("__BACKEND__", DECODE_BACKEND)
    )
    return HTMLResponse(page)

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "device": DEVICE,
        "decode_backend": DECODE_BACKEND,
        "prompt_length": PROMPT_LENGTH,
        "max_tokens": MAX_TOKENS,
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global g
    if g is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    dialog = [{"role": m.role, "content": m.content} for m in req.messages]
    assistant_header = g.tokenizer.encode_header({"role": "assistant", "content": ""})
    dialog_tokens = [g.tokenizer.encode_message(message)[0] for message in dialog]
    prompt_tokens = 1 + len(assistant_header) + sum(len(message_tokens) for message_tokens in dialog_tokens)
    while len(dialog) > 1:
        if prompt_tokens <= g.gen_args.prompt_length:
            break
        # Remove the oldest user/assistant pair or oldest message (keep system prompt if exists)
        if len(dialog) > 1 and dialog[0]["role"] == "system":
            prompt_tokens -= len(dialog_tokens.pop(1))
            dialog.pop(1)
        else:
            prompt_tokens -= len(dialog_tokens.pop(0))
            dialog.pop(0)

    if prompt_tokens > g.gen_args.prompt_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt uses {prompt_tokens} tokens but this server is compiled for "
                f"{g.gen_args.prompt_length}. The single message is too large."
            ),
        )

    tokens = encode_completion_dialog(g.tokenizer, dialog_tokens, assistant_header)
    requested_max_tokens = req.max_tokens or 512
    max_new_tokens = min(requested_max_tokens, g.gen_args.gen_length)

    stats, out_tokens = g.generate_all(
        [tokens],
        use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ,
        use_sampling=(req.temperature or 0.0) > 0.0,
        max_new_tokens=max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    answer_tokens = out_tokens[0]
    answer = g.tokenizer.decode(answer_tokens).strip()
    finish_reason = "stop"
    if len(answer_tokens) >= max_new_tokens and (not answer_tokens or answer_tokens[-1] != g.tokenizer.eos_id):
        finish_reason = "length"

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(answer_tokens),
            "total_tokens": prompt_tokens + len(answer_tokens)
        },
        "timing": {phase.name: phase.time for phase in stats.phases},
    }
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
