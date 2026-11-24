"""Motor de indicadores da PLD Intelligence.

Lê as quatro bases e devolve um dossiê por cliente. As regras estão no
catálogo em indicators.py. O motivo interno BLOCKED_BY_RISK não é lido
por nenhuma regra: se aparecer no dossiê, é só contexto operacional.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from .format import age_on, as_text, format_document, pretty_datetime
from .indicators import ADVERSE_KYC_REASONS, INDICATORS, level_of, score_of
from .report import describe

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PERIOD_END = date(2026, 4, 1)
CACHE_VERSION = "2026-11-23-b"
EFFECTIVE = {"complete", "approved"}
FLOW_TYPES = {"pix", "ted", "cash_withdrawal"}
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS


def load_frames(data_dir: Path | None = None) -> dict:
    folder = Path(data_dir) if data_dir else DATA_DIR
    users = _read(folder / "users.csv")
    accounts = _read(folder / "accounts.csv")
    kyc = _read(folder / "external_kyc.csv")
    tx = _read(folder / "transactions.csv")
    for frame, column in (
        (users, "income"),
        (users, "net_worth"),
        (accounts, "balance"),
        (tx, "amount"),
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    tx["created_at"] = pd.to_datetime(tx["created_at"])
    tx = tx.sort_values(["user_id", "created_at"], kind="mergesort").reset_index(drop=True)
    profiles = _profiles(users, accounts, kyc)
    groups = {user_id: group for user_id, group in tx.groupby("user_id", sort=False)}
    return {"users": users, "accounts": accounts, "kyc": kyc, "transactions": tx, "profiles": profiles, "groups": groups}


def load_analysis(data_dir: Path | None = None) -> tuple[dict, dict]:
    """Devolve bases e dossiês. Reaproveita o cache local se as regras não mudaram."""
    folder = Path(data_dir) if data_dir else DATA_DIR
    cache = folder / "analysis_cache.pkl"
    if cache.exists():
        try:
            with cache.open("rb") as handle:
                blob = pickle.load(handle)
            if blob.get("version") == CACHE_VERSION:
                return blob["frames"], blob["result"]
        except (OSError, pickle.UnpicklingError, KeyError, AttributeError):
            pass
    frames = load_frames(folder)
    result = detect(frames)
    try:
        with cache.open("wb") as handle:
            pickle.dump(
                {"version": CACHE_VERSION, "frames": frames, "result": result},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError:
        pass
    return frames, result


def detect(frames: dict) -> dict:
    profiles: dict[str, dict] = frames["profiles"]
    groups: dict[str, pd.DataFrame] = frames["groups"]
    cases: dict[str, dict] = {}
    for user_id, profile in profiles.items():
        cases[user_id] = _analyze_user(profile, groups.get(user_id), profiles)

    ranking = [case["user_id"] for case in cases.values() if case["findings"]]
    ranking.sort(key=lambda uid: (cases[uid]["score"], cases[uid]["metrics"]["ratio_7d"]), reverse=True)

    by_level: dict[str, int] = {}
    by_indicator: dict[str, int] = {code: 0 for code in INDICATORS}
    for user_id in ranking:
        case = cases[user_id]
        by_level[case["level"]] = by_level.get(case["level"], 0) + 1
        for finding in case["findings"]:
            by_indicator[finding["code"]] += 1

    tx = frames["transactions"]
    return {
        "cases": cases,
        "ranking": ranking,
        "hubs": _hubs(tx, profiles),
        "stats": {
            "n_users": len(profiles),
            "n_transactions": int(len(tx)),
            "n_alerts": len(ranking),
            "by_level": by_level,
            "by_indicator": by_indicator,
            "period": "01/01/2026 a 01/04/2026",
        },
    }


def suggest_case(result: dict) -> dict | None:
    """Caso com vários sinais de movimentação, útil para abrir direto na demonstração."""
    best: tuple[tuple, dict] | None = None
    for user_id in result["ranking"]:
        case = result["cases"][user_id]
        codes = {finding["code"] for finding in case["findings"]} - {"ADVERSE_KYC"}
        if len(codes) < 2:
            continue
        evidence = sum(len(finding["examples"]) for finding in case["findings"])
        key = (len(codes), case["score"], evidence)
        if best is None or key > best[0]:
            best = (key, case)
    if best:
        return best[1]
    if result["ranking"]:
        return result["cases"][result["ranking"][0]]
    return None


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _profiles(users: pd.DataFrame, accounts: pd.DataFrame, kyc: pd.DataFrame) -> dict[str, dict]:
    account_by_user = {row["user_id"]: row for row in accounts.to_dict("records")}
    kyc_by_user = {row["user_id"]: row for row in kyc.to_dict("records")}
    profiles = {}
    for row in users.to_dict("records"):
        user_id = row["user_id"]
        account = account_by_user.get(user_id, {})
        external = kyc_by_user.get(user_id, {})
        birth = as_text(row.get("birth_date"))
        age = age_on(birth, PERIOD_END) if birth else None
        document_type = row.get("document_type") or ""
        profiles[user_id] = {
            "user_id": user_id,
            "name": as_text(row.get("name")) or user_id,
            "document_type": document_type,
            "document": format_document(document_type, row.get("document_number")),
            "document_digits": "".join(ch for ch in (row.get("document_number") or "") if ch.isdigit()),
            "company_name": as_text(row.get("company_name")),
            "owner_cpf": format_document("CPF", row.get("owner_cpf")) if as_text(row.get("owner_cpf")) else None,
            "birth_label": pretty_datetime(birth, with_time=False) if birth else None,
            "age_label": f"{age} anos ao fim da base" if age is not None else None,
            "city": as_text(row.get("city")) or "cidade não informada",
            "state": as_text(row.get("state")) or "",
            "occupation": as_text(row.get("occupation")),
            "income": float(row.get("income") or 0),
            "net_worth": float(row.get("net_worth") or 0),
            "created_label": pretty_datetime(row.get("created_at")),
            "status": as_text(row.get("status")) or "não informado",
            "reason": as_text(row.get("reason")),
            "account_number": as_text(account.get("account_number")),
            "account_status": as_text(account.get("status")),
            "balance": float(account.get("balance") or 0),
            "kyc_status": as_text(external.get("status")),
            "kyc_reason": as_text(external.get("reason")),
            "kyc_updated": pretty_datetime(external.get("updated_at")),
        }
    return profiles


def _analyze_user(profile: dict, group: pd.DataFrame | None, profiles: dict[str, dict]) -> dict:
    if group is None or group.empty:
        effective = group if group is not None else pd.DataFrame()
        post_block = 0
    else:
        group = group.sort_values("created_at")
        effective = group[group["status"].isin(EFFECTIVE)]
        post_block = int(((group["status"] == "declined") & (group["status_code"] == "04")).sum())

    metrics = _metrics(effective, post_block, profile)
    facts = []
    for builder in (
        _cash_structuring,
        _pix_burst,
        _rapid_in_out,
        _high_volume,
        _unusual_hour,
        _many_originators,
        _repeated_external,
        _adverse_kyc,
    ):
        found = builder(effective, profile, profiles, metrics)
        if found:
            facts.append(found)

    findings = []
    for code, payload in facts:
        summary, narrative = describe(code, payload, profile)
        findings.append(
            {
                "code": code,
                "title": INDICATORS[code]["title"],
                "short": INDICATORS[code]["short"],
                "weight": INDICATORS[code]["weight"],
                "summary": summary,
                "narrative": narrative,
                "examples": payload.get("examples") or [],
                "facts": payload,
            }
        )
    codes = [finding["code"] for finding in findings]
    score = score_of(codes)
    return {
        "user_id": profile["user_id"],
        "profile": profile,
        "metrics": metrics,
        "findings": findings,
        "score": score,
        "level": level_of(score, len(findings)),
    }


def _metrics(effective: pd.DataFrame, post_block: int, profile: dict) -> dict:
    income = profile["income"] or 0.0
    if effective is None or effective.empty:
        return {
            "tx_count": 0,
            "credit_sum": 0.0,
            "debit_sum": 0.0,
            "volume_7d": 0.0,
            "ratio_7d": 0.0,
            "first_tx": "sem movimentação",
            "last_tx": "sem movimentação",
            "post_block_attempts": post_block,
        }
    credit = float(effective.loc[effective["transaction_direction"] == "credit", "amount"].sum())
    debit = float(effective.loc[effective["transaction_direction"] == "debit", "amount"].sum())
    debits = effective[effective["transaction_direction"] == "debit"]
    volume_7d, _, _ = _best_sum_window(debits, 7 * DAY_NS)
    return {
        "tx_count": int(len(effective)),
        "credit_sum": credit,
        "debit_sum": debit,
        "volume_7d": volume_7d,
        "ratio_7d": (volume_7d / income) if income else 0.0,
        "first_tx": pretty_datetime(effective["created_at"].iloc[0]),
        "last_tx": pretty_datetime(effective["created_at"].iloc[-1]),
        "post_block_attempts": post_block,
    }


def _cash_structuring(effective, profile, profiles, metrics):
    if effective is None or effective.empty:
        return None
    cash = effective[
        (effective["transaction_type"] == "cash_withdrawal") & (effective["transaction_direction"] == "debit")
    ]
    found = _best_count_window(cash, 6 * HOUR_NS, minimum=4, similar=True, min_total=2_000)
    if not found:
        return None
    window, start, end = found
    return "CASH_STRUCTURING", {
        "count": int(len(window)),
        "total": float(window["amount"].sum()),
        "hours": _hours(start, end),
        "start": pretty_datetime(start),
        "end": pretty_datetime(end),
        "examples": _examples(window, profiles),
    }


def _pix_burst(effective, profile, profiles, metrics):
    if effective is None or effective.empty:
        return None
    pix = effective[effective["transaction_type"] == "pix"]
    found = _best_count_window(pix, 3 * HOUR_NS, minimum=10)
    if not found:
        return None
    window, start, end = found
    return "MULTIPLE_PIX_SHORT_PERIOD", {
        "count": int(len(window)),
        "total": float(window["amount"].sum()),
        "hours": _hours(start, end),
        "start": pretty_datetime(start),
        "end": pretty_datetime(end),
        "examples": _examples(window, profiles),
    }


def _rapid_in_out(effective, profile, profiles, metrics):
    hit = _paired_outflow(effective, profile, profiles, within_ns=6 * HOUR_NS, min_ratio=0.75, min_multiple=0.5, min_amount=1_500)
    if not hit:
        return None
    return "RAPID_IN_OUT", hit


def _high_volume(effective, profile, profiles, metrics):
    payload = {}
    paired = _paired_outflow(
        effective, profile, profiles, within_ns=48 * HOUR_NS, min_ratio=0.50, min_multiple=8, min_amount=0
    )
    if paired:
        payload.update(paired)
    ratio = metrics["ratio_7d"]
    if ratio >= 3 and effective is not None and not effective.empty:
        debits = effective[effective["transaction_direction"] == "debit"]
        total, start, end = _best_sum_window(debits, 7 * DAY_NS)
        window = _slice_time(debits, start, end)
        payload.update(
            {
                "volume_7d": total,
                "ratio_7d": ratio,
                "window_start": pretty_datetime(start),
                "window_end": pretty_datetime(end),
            }
        )
        extra = _examples(window, profiles, limit=4)
        payload["examples"] = _merge_examples(payload.get("examples") or [], extra)
    if not payload:
        return None
    return "HIGH_VOLUME_TO_INCOME", payload


def _unusual_hour(effective, profile, profiles, metrics):
    income = profile["income"] or 0.0
    if income <= 0 or effective is None or effective.empty:
        return None
    hours = effective["created_at"].dt.hour
    night = effective[
        (effective["transaction_direction"] == "debit")
        & (effective["transaction_type"].isin(FLOW_TYPES))
        & (hours < 5)
        & (effective["amount"] >= 2 * income)
    ]
    if night.empty:
        return None
    total = float(night["amount"].sum())
    top = night.loc[night["amount"].idxmax()]
    return "UNUSUAL_HOUR", {
        "count": int(len(night)),
        "total": total,
        "max_amount": float(top["amount"]),
        "max_ratio": float(top["amount"]) / income,
        "max_when": pretty_datetime(top["created_at"]),
        "examples": _examples(night, profiles),
    }


def _many_originators(effective, profile, profiles, metrics):
    if effective is None or effective.empty:
        return None
    credits = effective[
        (effective["transaction_direction"] == "credit")
        & (effective["transaction_type"].isin(["pix", "ted"]))
        & (effective["counterparty_user_id"].map(lambda value: as_text(value) is None))
        & (effective["counterparty_account_number"].map(lambda value: as_text(value) is not None))
    ]
    found = _best_distinct_window(credits, 72 * HOUR_NS, "counterparty_account_number", minimum=12)
    if not found:
        return None
    window, start, end, distinct = found
    return "MANY_ORIGINATORS", {
        "count": int(len(window)),
        "originators": int(distinct),
        "total": float(window["amount"].sum()),
        "hours": _hours(start, end),
        "start": pretty_datetime(start),
        "end": pretty_datetime(end),
        "examples": _examples(window, profiles),
    }


def _repeated_external(effective, profile, profiles, metrics):
    income = profile["income"] or 0.0
    if effective is None or effective.empty:
        return None
    threshold = max(1_500.0, 0.35 * income)
    debits = effective[
        (effective["transaction_direction"] == "debit")
        & (effective["transaction_type"].isin(["pix", "ted"]))
        & (effective["counterparty_user_id"].map(lambda value: as_text(value) is None))
        & (effective["counterparty_account_number"].map(lambda value: as_text(value) is not None))
    ]
    best = None
    for account, group in debits.groupby("counterparty_account_number", sort=False):
        found = _best_count_window(group, 5 * DAY_NS, minimum=4, min_total=threshold)
        if not found:
            continue
        window, start, end = found
        key = (float(window["amount"].sum()), int(len(window)))
        if best is None or key > best[0]:
            best = (key, window, start, end, account)
    if not best:
        return None
    _, window, start, end, account = best
    return "REPEATED_EXTERNAL", {
        "count": int(len(window)),
        "total": float(window["amount"].sum()),
        "hours": _hours(start, end),
        "start": pretty_datetime(start),
        "end": pretty_datetime(end),
        "account": account,
        "examples": _examples(window, profiles),
    }


def _adverse_kyc(effective, profile, profiles, metrics):
    if profile.get("kyc_status") == "blocked" and profile.get("kyc_reason") in ADVERSE_KYC_REASONS:
        return "ADVERSE_KYC", {"reason": profile["kyc_reason"], "updated": profile.get("kyc_updated"), "examples": []}
    return None


def _paired_outflow(effective, profile, profiles, within_ns: int, min_ratio: float, min_multiple: float, min_amount: float):
    income = profile["income"] or 0.0
    if income <= 0 or effective is None or effective.empty:
        return None
    floor = max(min_amount, min_multiple * income)
    credits = effective[
        (effective["transaction_direction"] == "credit") & (effective["transaction_type"].isin(["pix", "ted"]))
    ]
    debits = effective[
        (effective["transaction_direction"] == "debit") & (effective["transaction_type"].isin(FLOW_TYPES))
    ]
    if credits.empty or debits.empty:
        return None
    debit_ns = debits["created_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    debit_amount = debits["amount"].to_numpy(dtype="float64")
    best = None
    for row in credits.itertuples(index=False):
        amount = float(row.amount)
        if amount < floor:
            continue
        start = pd.Timestamp(row.created_at).to_datetime64().astype("int64")
        end = start + within_ns
        left = int(debit_ns.searchsorted(start, side="right"))
        right = int(debit_ns.searchsorted(end, side="right"))
        if left >= right:
            continue
        total = float(debit_amount[left:right].sum())
        if total < min_ratio * amount:
            continue
        key = amount
        if best is None or key > best[0]:
            best = (key, row, total, left, right, end)
    if not best:
        return None
    _, row, total, left, right, _end_ns = best
    window = debits.iloc[left:right]
    credit_row = credits[credits["created_at"] == row.created_at].iloc[0]
    return {
        "credit_amount": float(credit_row["amount"]),
        "credit_ratio": float(credit_row["amount"]) / income,
        "credit_when": pretty_datetime(credit_row["created_at"]),
        "credit_type": credit_row["transaction_type"],
        "credit_detail": _detail(credit_row, profiles),
        "out_total": total,
        "out_pct": int(round(100 * total / float(credit_row["amount"]))),
        "out_hours": _hours(credit_row["created_at"], window["created_at"].iloc[-1]),
        "out_count": int(len(window)),
        "examples": _merge_examples(_examples(credit_row.to_frame().T, profiles, limit=1), _examples(window, profiles, limit=4)),
    }


def _best_count_window(frame: pd.DataFrame, window_ns: int, minimum: int, similar: bool = False, min_total: float = 0.0):
    if frame is None or len(frame) < minimum:
        return None
    frame = frame.sort_values("created_at")
    times = frame["created_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    amounts = frame["amount"].to_numpy(dtype="float64")
    best = None
    n = len(frame)
    for i in range(n):
        for j in range(i + minimum - 1, n):
            if times[j] - times[i] > window_ns:
                break
            total = float(amounts[i : j + 1].sum())
            if total < min_total:
                continue
            if similar:
                chunk = amounts[i : j + 1]
                mean = float(chunk.mean())
                if mean <= 0 or (float(chunk.max()) - float(chunk.min())) / mean > 0.35:
                    continue
            key = (j - i + 1, total)
            if best is None or key > best[0]:
                best = (key, i, j)
    if not best:
        return None
    _, i, j = best
    window = frame.iloc[i : j + 1]
    return window, window["created_at"].iloc[0], window["created_at"].iloc[-1]


def _best_distinct_window(frame: pd.DataFrame, window_ns: int, column: str, minimum: int):
    if frame is None or frame.empty:
        return None
    frame = frame.sort_values("created_at")
    times = frame["created_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    keys = frame[column].astype(str).to_numpy()
    best = None
    n = len(frame)
    for i in range(n):
        seen: set[str] = set()
        for j in range(i, n):
            if times[j] - times[i] > window_ns:
                break
            seen.add(keys[j])
            if len(seen) < minimum:
                continue
            key = (len(seen), j - i + 1)
            if best is None or key > best[0]:
                best = (key, i, j, len(seen))
    if not best:
        return None
    _, i, j, distinct = best
    window = frame.iloc[i : j + 1]
    return window, window["created_at"].iloc[0], window["created_at"].iloc[-1], distinct


def _best_sum_window(frame: pd.DataFrame, window_ns: int) -> tuple[float, object, object]:
    if frame is None or frame.empty:
        return 0.0, None, None
    frame = frame.sort_values("created_at")
    times = frame["created_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    amounts = frame["amount"].to_numpy(dtype="float64")
    best = 0.0
    best_i = 0
    best_j = 0
    running = 0.0
    j = 0
    for i in range(len(frame)):
        running += amounts[i]
        while times[i] - times[j] > window_ns:
            running -= amounts[j]
            j += 1
        if running > best:
            best = float(running)
            best_i = j
            best_j = i
    return best, frame["created_at"].iloc[best_i], frame["created_at"].iloc[best_j]


def _slice_time(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None or frame.empty:
        return frame.iloc[0:0]
    mask = (frame["created_at"] >= start) & (frame["created_at"] <= end)
    return frame.loc[mask]


def _hours(start, end) -> float:
    delta = pd.Timestamp(end) - pd.Timestamp(start)
    return max(delta.total_seconds() / 3600, 0.0)


def _examples(frame: pd.DataFrame, profiles: dict[str, dict], limit: int = 6) -> list[dict]:
    if frame is None or frame.empty:
        return []
    ordered = frame.sort_values("created_at")
    if len(ordered) > limit:
        ordered = pd.concat([ordered.head(limit // 2), ordered.tail(limit - limit // 2)])
        ordered = ordered.drop_duplicates(subset=["created_at", "amount", "transaction_type"])
    rows = []
    for record in ordered.to_dict("records"):
        rows.append(
            {
                "when": pretty_datetime(record["created_at"]),
                "type": record["transaction_type"],
                "direction": record["transaction_direction"],
                "amount": float(record["amount"]),
                "detail": _detail(record, profiles),
            }
        )
    return rows


def _merge_examples(left: list[dict], right: list[dict]) -> list[dict]:
    seen = set()
    merged = []
    for item in left + right:
        key = (item["when"], item["type"], item["direction"], round(item["amount"], 2))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[:8]


def _detail(record, profiles: dict[str, dict]) -> str:
    if not isinstance(record, dict):
        record = dict(record)
    cp_user = as_text(record.get("counterparty_user_id"))
    cp_account = as_text(record.get("counterparty_account_number"))
    if cp_user:
        other = profiles.get(cp_user)
        if other:
            occupation = other.get("occupation")
            who = other["name"] + (f", {occupation}" if occupation else "")
            account = cp_account or "conta não informada"
            return f"contraparte interna {who}, conta {account}"
        return f"contraparte interna {cp_user}"
    if cp_account:
        return f"conta externa {cp_account}"
    if record.get("transaction_type") == "cash_withdrawal":
        return "saque em espécie, sem identificação do destino"
    if record.get("transaction_type") == "card":
        return "compra em cartão"
    return "contraparte não identificada"


def _hubs(tx: pd.DataFrame, profiles: dict[str, dict]) -> list[dict]:
    external = tx[
        tx["status"].isin(EFFECTIVE)
        & (tx["transaction_direction"] == "debit")
        & tx["transaction_type"].isin(["pix", "ted"])
        & (tx["counterparty_user_id"].map(lambda value: as_text(value) is None))
        & (tx["counterparty_account_number"].map(lambda value: as_text(value) is not None))
    ]
    if external.empty:
        return []
    hubs = []
    for account, group in external.groupby("counterparty_account_number", sort=False):
        users = group["user_id"].drop_duplicates()
        if len(users) < 3:
            continue
        sample = []
        for user_id in users.head(4):
            profile = profiles.get(user_id)
            sample.append(profile["name"] if profile else user_id)
        hubs.append(
            {
                "account": account,
                "n_users": int(len(users)),
                "n_tx": int(len(group)),
                "total": float(group["amount"].sum()),
                "sample": sample,
            }
        )
    hubs.sort(key=lambda item: (item["n_users"], item["total"]), reverse=True)
    return hubs[:8]
