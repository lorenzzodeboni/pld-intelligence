"""Decisões do analista e fila de demonstração. Arquivos locais, fora do cálculo do score."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DECISIONS_PATH = DATA_DIR / "analyst_decisions.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.json"

DECISION_LABEL = {
    "em_analise": "Em análise",
    "comunicar": "Encaminhado para comunicação",
    "encerrar": "Encerrado sem comunicação",
    "falso_positivo": "Falso positivo",
}


def read_decisions() -> dict[str, dict]:
    if not DECISIONS_PATH.exists():
        return {}
    frame = pd.read_csv(DECISIONS_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    latest: dict[str, dict] = {}
    for row in frame.to_dict("records"):
        latest[row["user_id"]] = row
    return latest


def write_decision(user_id: str, decision: str, score: int, indicators: str, note: str = "") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "decided_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "decision": decision,
        "score": str(score),
        "indicators": indicators,
        "note": note.strip(),
    }
    frame = pd.DataFrame([row])
    frame.to_csv(
        DECISIONS_PATH,
        mode="a",
        header=not DECISIONS_PATH.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def read_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    return json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))


def add_feedback(user_id: str, codes: list[str], note: str = "") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_feedback()
    rows.append(
        {
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "codes": codes,
            "note": note.strip(),
        }
    )
    FEEDBACK_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_demo_state() -> None:
    if DECISIONS_PATH.exists():
        DECISIONS_PATH.unlink()
    if FEEDBACK_PATH.exists():
        FEEDBACK_PATH.unlink()
