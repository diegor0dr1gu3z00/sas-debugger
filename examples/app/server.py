"""Local demo web app — watch the agent debug a deep discrepancy in real time.

Run:
    .venv/bin/python examples/app/server.py
Then open http://127.0.0.1:8000

The page lets you load the two corresponding `.egp` projects and the `.xlsx` of
suspect cycles (or use the bundled sample assets), then streams the agent's
debug trace — reasoning, key-table extractions, diagnostic SQL, oracle verdicts
and the final localisation — as Server-Sent Events.

By default the trace is deterministic (a curated expert pass). To drive it with
a real small LM, set the backend before launching:

    # in-process transformers (base + optional LoRA adapter):
    INFER_BACKEND=transformers INFER_ADAPTER=checkpoints/sft-0.5b-ext/lora \
        .venv/bin/python examples/app/server.py

    # llama-server HTTP endpoint:
    INFER_BACKEND=llama.cpp INFER_LLAMA_URL=http://127.0.0.1:8080 \
        .venv/bin/python examples/app/server.py

The chosen backend is exported in the SSE header so the UI can show it.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "app_assets"

# sample assets the server auto-loads if the user uploads nothing
SAMPLE_SRC = ASSETS / "src_basilea.egp"
SAMPLE_REP = ASSETS / "rep_lgd.egp"
SAMPLE_XLS = ASSETS / "ciclos_sospechosos.xlsx"

app = FastAPI(title="SAS Reconcile SLM — debug demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# in-memory session store: session_id -> dict(file paths, summary)
sessions: dict[str, dict] = {}

INDEX_HTML = (HERE / "index.html").read_text(encoding="utf-8")

# ── Optional real inference backend (off -> deterministic trace) ─────────────
INFER_BACKEND = os.environ.get("INFER_BACKEND", "").strip().lower()
INFER_MODEL = None  # lazily built generator: generate(system, user) -> str


def _inference_generator():
    """Build (and cache) the LM generator from env, or None for deterministic."""
    global INFER_MODEL
    if INFER_MODEL is not None or not INFER_BACKEND:
        return INFER_MODEL
    import sys
    sys.path.insert(0, str(HERE))
    from inference import make_generator
    INFER_MODEL = make_generator(
        backend=INFER_BACKEND,
        base_model=os.environ.get("INFER_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        adapter=os.environ.get("INFER_ADAPTER") or None,
        llama_url=os.environ.get("INFER_LLAMA_URL") or None,
        device=os.environ.get("INFER_DEVICE", "auto"),
        dtype=os.environ.get("INFER_DTYPE", "auto"),
        max_new_tokens=int(os.environ.get("INFER_MAX_NEW_TOKENS", "768")),
    )
    return INFER_MODEL


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/assets")
def assets() -> dict:
    return {"src": SAMPLE_SRC.name, "rep": SAMPLE_REP.name,
            "xlsx": SAMPLE_XLS.name}


@app.post("/api/session")
async def create_session(
    src: UploadFile | None = File(None),
    rep: UploadFile | None = File(None),
    xlsx: UploadFile | None = File(None),
    prompt: str = Form(""),
) -> dict:
    sid = uuid.uuid4().hex
    tmp = Path("/tmp/sas-slm-demo") / sid
    tmp.mkdir(parents=True, exist_ok=True)

    def save(upload: UploadFile | None, fallback: Path, default: Path) -> tuple[Path, str]:
        if upload is not None:
            p = tmp / (upload.filename or fallback.name)
            p.write_bytes(upload.file.read())
            return p, upload.filename or p.name
        return default, default.name  # use bundled sample when nothing uploaded

    src_p, src_n = save(src, SAMPLE_SRC, SAMPLE_SRC)
    rep_p, rep_n = save(rep, SAMPLE_REP, SAMPLE_REP)
    xlsx_p, xlsx_n = save(xlsx, SAMPLE_XLS, SAMPLE_XLS)
    sessions[sid] = {"src": str(src_p), "rep": str(rep_p), "xlsx": str(xlsx_p),
                     "prompt": (prompt or "").strip(),
                     "names": {"src": src_n, "rep": rep_n, "xlsx": xlsx_n}}
    return {"session_id": sid, "src": src_n, "rep": rep_n, "xlsx": xlsx_n}


@app.get("/api/debug/{sid}")
def debug_stream(sid: str) -> StreamingResponse:
    import sys
    sys.path.insert(0, str(HERE))
    from agent import run_agent

    sess = sessions[sid]
    model = _inference_generator()

    def gen():
        for ev in run_agent(sess["src"], sess["rep"], sess["xlsx"], model,
                            prompt=sess.get("prompt", ""),
                            names=sess.get("names", {})):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/api/infer")
def infer_status() -> dict:
    return {"active": bool(INFER_BACKEND), "backend": INFER_BACKEND or "deterministic"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8010"))
    print(f"Open http://127.0.0.1:{port}  (sample assets: {SAMPLE_SRC.name}, "
          f"{SAMPLE_REP.name}, {SAMPLE_XLS.name})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
