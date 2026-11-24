"""Checagens do motor e um resumo da base sintética.

Uso, a partir da pasta do projeto:
  .venv\\Scripts\\python.exe src\\pld\\evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pld.engine import detect, load_frames  # noqa: E402
from pld.indicators import INDICATORS  # noqa: E402


def _person(user_id: str, income: float = 3000.0, reason: str = "", status: str = "active", kyc_status: str = "approved", kyc_reason: str = "") -> tuple[dict, dict, dict]:
    user = {
        "user_id": user_id,
        "document_type": "CPF",
        "document_number": "12345678901",
        "name": "Cliente Teste",
        "company_name": "",
        "owner_cpf": "",
        "created_at": "2026-01-05 10:00:00",
        "updated_at": "2026-01-05 10:00:00",
        "status": status,
        "reason": reason,
        "birth_date": "1990-01-01",
        "address": "Rua 1",
        "city": "São Paulo",
        "state": "SP",
        "country": "Brasil",
        "occupation": "Analista",
        "income": income,
        "net_worth": 10000.0,
    }
    account = {
        "account_id": "A" + user_id[1:],
        "account_number": "0001-00000001-0",
        "user_id": user_id,
        "balance": 1000.0,
        "created_at": "2026-01-05 10:01:00",
        "updated_at": "2026-02-01 10:00:00",
        "status": "active",
    }
    kyc = {
        "user_id": user_id,
        "created_at": "2026-01-05 10:00:00",
        "updated_at": "2026-01-05 11:00:00",
        "status": kyc_status,
        "reason": kyc_reason,
    }
    return user, account, kyc


def _tx(user_id: str, when: str, kind: str, direction: str, amount: float, account: str = "", cp_user: str = "", status: str = "complete", code: str = "00") -> dict:
    return {
        "transaction_id": when,
        "user_id": user_id,
        "counterparty_user_id": cp_user,
        "counterparty_account_number": account,
        "created_at": pd.Timestamp(when),
        "updated_at": when,
        "transaction_type": kind,
        "transaction_direction": direction,
        "amount": amount,
        "status": status,
        "status_code": code,
        "reason": "",
    }


def _run(rows: list[dict], user: dict, account: dict, kyc: dict) -> dict:
    from pld.engine import _profiles

    users = pd.DataFrame([user])
    accounts = pd.DataFrame([account])
    kyc_frame = pd.DataFrame([kyc])
    tx = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["user_id", "created_at", "status", "status_code"])
    profiles = _profiles(users, accounts, kyc_frame)
    groups = {user["user_id"]: tx} if rows else {}
    return detect({"profiles": profiles, "groups": groups, "transactions": tx if rows else pd.DataFrame(columns=["user_id", "status", "transaction_direction", "transaction_type", "counterparty_user_id", "counterparty_account_number", "amount"])})


def _codes(result: dict) -> set[str]:
    case = next(iter(result["cases"].values()))
    return {finding["code"] for finding in case["findings"]}


def self_check() -> None:
    user, account, kyc = _person("U00001")
    base = "2026-02-10 12:00:00"

    calm = [_tx("U00001", f"2026-02-{day:02d} 11:00:00", "pix", "debit", 80, account=f"9999-0000000{day}-0") for day in range(1, 6)]
    assert _codes(_run(calm, user, account, kyc)) == set()

    cards = [_tx("U00001", f"2026-02-10 12:{minute:02d}:00", "card", "debit", 50) for minute in range(0, 12)]
    assert "MULTIPLE_PIX_SHORT_PERIOD" not in _codes(_run(cards, user, account, kyc))

    burst = [_tx("U00001", f"2026-02-10 15:{minute:02d}:00", "pix", "debit", 40, account=f"8888-{minute:08d}-0") for minute in range(0, 10)]
    assert "MULTIPLE_PIX_SHORT_PERIOD" in _codes(_run(burst, user, account, kyc))

    cash = [_tx("U00001", f"2026-02-11 16:{10 + minute:02d}:00", "cash_withdrawal", "debit", 1000) for minute in range(0, 6)]
    assert "CASH_STRUCTURING" in _codes(_run(cash, user, account, kyc))

    rapid = [
        _tx("U00001", "2026-02-12 09:00:00", "pix", "credit", 10000, account="7777-00000001-0"),
        _tx("U00001", "2026-02-12 10:00:00", "pix", "debit", 8000, account="7777-00000002-0"),
    ]
    rapid_codes = _codes(_run(rapid, user, account, kyc))
    assert "RAPID_IN_OUT" in rapid_codes
    assert "HIGH_VOLUME_TO_INCOME" not in rapid_codes

    parked = [
        _tx("U00001", "2026-02-01 09:00:00", "ted", "credit", 4000, account="7777-00000001-0"),
        _tx("U00001", "2026-02-18 09:00:00", "ted", "credit", 100000, account="7777-00000009-0"),
    ]
    assert "HIGH_VOLUME_TO_INCOME" not in _codes(_run(parked, user, account, kyc))

    incompatible = [
        _tx("U00001", "2026-02-01 09:00:00", "ted", "credit", 1000, account="7777-00000001-0"),
        _tx("U00001", "2026-02-18 09:00:00", "ted", "credit", 30000, account="7777-00000009-0"),
        _tx("U00001", "2026-02-18 19:00:00", "pix", "debit", 20000, account="7777-00000008-0"),
    ]
    incompatible_codes = _codes(_run(incompatible, user, account, kyc))
    assert "HIGH_VOLUME_TO_INCOME" in incompatible_codes
    assert "RAPID_IN_OUT" not in incompatible_codes

    opening = [_tx("U00001", "2026-02-01 09:00:00", "ted", "credit", 100000, account="7777-00000001-0")]
    assert "HIGH_VOLUME_TO_INCOME" not in _codes(_run(opening, user, account, kyc))

    night = [_tx("U00001", "2026-02-14 01:20:00", "pix", "debit", 7000, account="7777-00000003-0")]
    night_codes = _codes(_run(night, user, account, kyc))
    assert "UNUSUAL_HOUR" in night_codes
    assert "HIGH_VOLUME_TO_INCOME" not in night_codes

    origins = [
        _tx("U00001", f"2026-02-16 {hour:02d}:10:00", "pix", "credit", 200, account=f"5555-{index:08d}-0")
        for index, hour in enumerate(range(8, 20))
    ]
    assert "MANY_ORIGINATORS" in _codes(_run(origins, user, account, kyc))

    repeated = [
        _tx("U00001", f"2026-02-{14 + index} 11:00:00", "pix", "debit", 800, account="4444-00000001-9")
        for index in range(4)
    ]
    assert "REPEATED_EXTERNAL" in _codes(_run(repeated, user, account, kyc))

    blocked_user, blocked_account, blocked_kyc = _person("U00001", reason="BLOCKED_BY_RISK", status="blocked")
    blocked = [_tx("U00001", base, "card", "debit", 30)]
    assert _codes(_run(blocked, blocked_user, blocked_account, blocked_kyc)) == set()

    sanctioned_user, sanctioned_account, sanctioned_kyc = _person("U00001", kyc_status="blocked", kyc_reason="SANCTIONS_MATCH")
    sanctioned = _run([], sanctioned_user, sanctioned_account, sanctioned_kyc)
    case = sanctioned["cases"]["U00001"]
    assert _codes(sanctioned) == {"ADVERSE_KYC"}
    assert case["score"] == INDICATORS["ADVERSE_KYC"]["weight"]
    assert case["level"] == "alto"
    print(f"autoconsistência: ok ({case['score']} no KYC de sanção, sem usar bloqueio interno)")


def summarize() -> None:
    started = time.perf_counter()
    frames = load_frames()
    result = detect(frames)
    elapsed = time.perf_counter() - started
    stats = result["stats"]
    print(f"base: {stats['n_users']} clientes, {stats['n_transactions']} transações, {elapsed:.1f}s")
    print(f"alertas: {stats['n_alerts']} | níveis: {stats['by_level']}")
    for code, count in stats["by_indicator"].items():
        print(f"  {code}: {count}")
    ratios = [case["metrics"]["ratio_7d"] for case in result["cases"].values()]
    series = pd.Series(ratios)
    print("clientes com débito-7d / renda >= 3, 5, 8:", int((series >= 3).sum()), int((series >= 5).sum()), int((series >= 8).sum()))
    print("razão débito-7d / renda — p50 {:.2f} p90 {:.2f} p99 {:.2f} max {:.2f}".format(
        series.quantile(0.50), series.quantile(0.90), series.quantile(0.99), series.max()
    ))
    print(f"destinos compartilhados: {len(result['hubs'])}")
    for hub in result["hubs"][:5]:
        print(f"  {hub['account']}: {hub['n_users']} clientes, R$ {hub['total']:.2f}")
    if result["ranking"]:
        top = result["cases"][result["ranking"][0]]
        print(f"topo da fila: {top['profile']['name']} {top['user_id']} score {top['score']}")
        for finding in top["findings"]:
            print(f"  - {finding['code']}: {finding['summary']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    self_check()
    summarize()
