"""Formatação de valores para a interface e para os relatórios."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

MISSING = {"", "nan", "none", "<na>", "nat"}


def is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if value is pd.NA or pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in MISSING


def as_text(value) -> str | None:
    if is_blank(value):
        return None
    return str(value).strip()


def brl(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "R$ 0,00"
    formatted = f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def ratio_label(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0"
    return f"{float(value):.1f}".replace(".", ",")


def duration_label(hours: float) -> str:
    if hours < 1:
        minutes = max(1, int(round(hours * 60)))
        return f"{minutes} minuto" if minutes == 1 else f"{minutes} minutos"
    text = f"{hours:.1f}".replace(".", ",")
    return f"{text} hora" if hours < 1.05 else f"{text} horas"


def format_document(document_type: str | None, number: str | None) -> str:
    digits = "".join(ch for ch in (number or "") if ch.isdigit())
    if document_type == "CPF" and len(digits) == 11:
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
    if document_type == "CNPJ" and len(digits) == 14:
        return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
    return digits or "não informado"


def pretty_datetime(value, with_time: bool = True) -> str:
    if is_blank(value):
        return "não informado"
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "não informado"
    fmt = "%d/%m/%Y %H:%M" if with_time else "%d/%m/%Y"
    return ts.strftime(fmt)


def age_on(birth, reference: date) -> int | None:
    if is_blank(birth):
        return None
    born = pd.Timestamp(birth)
    if pd.isna(born):
        return None
    born_d = born.date()
    return reference.year - born_d.year - ((reference.month, reference.day) < (born_d.month, born_d.day))


def status_label(value: str | None) -> str:
    labels = {
        "active": "ativa",
        "blocked": "bloqueada",
        "inactive": "inativa",
        "approved": "aprovado",
        "declined": "recusado",
        "awaiting": "em processamento",
    }
    if is_blank(value):
        return "não informada"
    return labels.get(str(value), str(value))


def parse_datetime(value) -> datetime | None:
    if is_blank(value):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()
