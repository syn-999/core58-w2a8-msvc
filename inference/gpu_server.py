import os
import time
import uuid
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from generate import FastGen, GenArgs
from tokenizer import ChatFormat

app = FastAPI()

# Global config
CKPT_DIR = "./checkpoints/"
DEVICE = "cuda:0"

g = None

@app.on_event("startup")
def load_model():
    global g
    print(f"Loading model on {DEVICE} from {CKPT_DIR}")
    torch.cuda.set_device(DEVICE)
    g = FastGen.build(CKPT_DIR, GenArgs(), DEVICE)
    # Wrap tokenizer in ChatFormat for chat/completions
    g.tokenizer = ChatFormat(g.tokenizer)
    print("Model loaded and ready for inference.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "bitnet"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global g
    
    # We only process batch size 1 natively for now based on generator constraints
    dialog = [{"role": m.role, "content": m.content} for m in req.messages]
    
    # Tokenize prompt using the ChatFormat structure
    tokens = [g.tokenizer.encode_dialog_prompt(dialog=dialog, completion=True)]
    
    # Run the BitNet generator
    stats, out_tokens = g.generate_all(
        tokens,
        use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ,
        use_sampling=(req.temperature > 0.0)
    )
    
    # Decode the result (batch index 0)
    answer = g.tokenizer.decode(out_tokens[0])
    
    # Build OpenAI compatible response
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
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokens[0]),
            "completion_tokens": len(out_tokens[0]),
            "total_tokens": len(tokens[0]) + len(out_tokens[0])
        }
    }
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
