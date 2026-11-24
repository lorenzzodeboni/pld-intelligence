"""Geração da base de usuários (pessoas físicas e jurídicas fictícias)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import DAY, END_S, HOUR, N_USERS, START, USER_COLUMNS, to_datetime_str, to_money

CNPJ_SHARE = 0.15
UNDERAGE_SHARE = 0.012
CADASTRAL_UPDATE_SHARE = 0.25
LAST_SIGNUP_DAY = 83  # 2026-03-25

CITIES = [
    ("São Paulo", "SP", 11.4), ("Rio de Janeiro", "RJ", 6.2), ("Brasília", "DF", 2.8),
    ("Fortaleza", "CE", 2.4), ("Salvador", "BA", 2.4), ("Belo Horizonte", "MG", 2.3),
    ("Manaus", "AM", 2.1), ("Curitiba", "PR", 1.8), ("Recife", "PE", 1.5),
    ("Goiânia", "GO", 1.4), ("Porto Alegre", "RS", 1.3), ("Belém", "PA", 1.3),
    ("Guarulhos", "SP", 1.3), ("Campinas", "SP", 1.1), ("São Luís", "MA", 1.0),
    ("Maceió", "AL", 0.96), ("Campo Grande", "MS", 0.9), ("Teresina", "PI", 0.87),
    ("João Pessoa", "PB", 0.83), ("Natal", "RN", 0.75), ("Uberlândia", "MG", 0.71),
    ("Ribeirão Preto", "SP", 0.7), ("Cuiabá", "MT", 0.65), ("Joinville", "SC", 0.62),
    ("Aracaju", "SE", 0.6), ("Londrina", "PR", 0.56), ("Florianópolis", "SC", 0.54),
    ("Niterói", "RJ", 0.48), ("Porto Velho", "RO", 0.46), ("Macapá", "AP", 0.44),
    ("Santos", "SP", 0.42), ("Boa Vista", "RR", 0.41), ("Rio Branco", "AC", 0.36),
    ("Vitória", "ES", 0.32), ("Palmas", "TO", 0.3),
]

OCCUPATIONS_LOW = [
    "Atendente", "Auxiliar administrativo", "Vendedor", "Motorista", "Operador de caixa",
    "Recepcionista", "Cozinheiro", "Auxiliar de produção", "Técnico de enfermagem", "Entregador",
]
OCCUPATIONS_MID = [
    "Analista administrativo", "Professor", "Enfermeiro", "Técnico em TI", "Contador",
    "Autônomo", "Corretor de imóveis", "Analista financeiro", "Designer", "Supervisor de vendas",
]
OCCUPATIONS_HIGH = [
    "Engenheiro", "Médico", "Advogado", "Gerente", "Desenvolvedor de software",
    "Empresário", "Diretor", "Dentista", "Arquiteto", "Consultor",
]
CNPJ_OCCUPATIONS = ["Sócio-administrador", "Empresário"]

SIGNUP_HOUR_WEIGHTS = np.array([
    0.2, 0.1, 0.05, 0.05, 0.05, 0.2, 0.6, 1.5, 3.0, 4.5, 5.0, 5.0,
    4.5, 4.5, 4.8, 4.8, 4.8, 4.8, 5.0, 5.5, 5.5, 4.5, 3.0, 1.0,
])


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _birth_date(created: date, age: int, days_back: int) -> date:
    try:
        anniversary = created.replace(year=created.year - age)
    except ValueError:
        anniversary = created.replace(year=created.year - age, day=28)
    return anniversary - timedelta(days=days_back)


def age_at(birth: date, reference: date) -> int:
    return reference.year - birth.year - ((reference.month, reference.day) < (birth.month, birth.day))


def _unique_company(fake) -> str:
    try:
        return fake.unique.company()
    except Exception:
        return f"{fake.last_name()} {fake.last_name()} Ltda."


def generate_users(rng, rnd, fake) -> pd.DataFrame:
    """Gera os atributos cadastrais. Status, reason e updated_at são definidos em finalize_users."""
    n = N_USERS
    is_cnpj = rng.random(n) < CNPJ_SHARE

    days = np.arange(LAST_SIGNUP_DAY + 1)
    day_p = np.exp(-days / 35.0)
    day_p /= day_p.sum()
    created_s = (
        rng.choice(days, n, p=day_p) * DAY
        + rng.choice(24, n, p=SIGNUP_HOUR_WEIGHTS / SIGNUP_HOUR_WEIGHTS.sum()) * HOUR
        + rng.integers(0, HOUR, n)
    )
    created_s = np.sort(created_s)

    city_idx = rng.choice(len(CITIES), n, p=np.array([c[2] for c in CITIES]) / sum(c[2] for c in CITIES))

    rows = []
    for k in range(n):
        created_dt = START + pd.Timedelta(seconds=int(created_s[k]))
        city, state, _ = CITIES[city_idx[k]]
        address = f"{fake.street_address()}, {fake.bairro()}"
        name = f"{fake.first_name()} {fake.last_name()}" + (f" {fake.last_name()}" if rnd.random() < 0.6 else "")
        row = {
            "user_id": f"U{k + 1:05d}",
            "name": name,
            "address": address,
            "city": city,
            "state": state,
            "country": "Brasil",
            "created_s": int(created_s[k]),
        }
        if is_cnpj[k]:
            revenue = float(np.clip(np.exp(rng.normal(np.log(60_000), 1.0)), 8_000, 3_000_000))
            row.update({
                "document_type": "CNPJ",
                "document_number": _digits(fake.unique.cnpj()),
                "company_name": _unique_company(fake),
                "owner_cpf": _digits(fake.unique.cpf()),
                "birth_date": None,
                "occupation": rnd.choice(CNPJ_OCCUPATIONS) if rnd.random() < 0.5 else None,
                "income": revenue,
                "net_worth": revenue * float(np.exp(rng.normal(np.log(4.0), 0.7))),
            })
        else:
            underage = rng.random() < UNDERAGE_SHARE
            age = int(rng.integers(15, 18)) if underage else int(np.clip(rng.normal(39, 13), 18, 85))
            birth = _birth_date(created_dt.date(), age, rnd.randint(1, 360))
            if underage:
                income = float(rng.uniform(300, 1_500))
                net_worth = float(rng.uniform(0, 3_000))
                occupation = "Estudante"
            else:
                income = float(np.clip(np.exp(rng.normal(np.log(3_500), 0.75)), 1_300, 150_000))
                age_factor = 0.3 + (age - 18) / 15.0
                net_worth = income * 12 * float(np.exp(rng.normal(0.0, 0.9))) * age_factor
                if age >= 65 and rnd.random() < 0.7:
                    occupation = "Aposentado"
                elif income < 3_000:
                    occupation = rnd.choice(OCCUPATIONS_LOW)
                elif income < 9_000:
                    occupation = rnd.choice(OCCUPATIONS_MID)
                else:
                    occupation = rnd.choice(OCCUPATIONS_HIGH)
            row.update({
                "document_type": "CPF",
                "document_number": _digits(fake.unique.cpf()),
                "company_name": None,
                "owner_cpf": None,
                "birth_date": birth.isoformat(),
                "occupation": occupation,
                "income": income,
                "net_worth": net_worth,
            })
        row["income"] = int(round(row["income"] * 100))
        row["net_worth"] = int(round(row["net_worth"] * 100))
        rows.append(row)

    return pd.DataFrame(rows)


def finalize_users(users: pd.DataFrame, kyc: pd.DataFrame, risk_block: dict, rng) -> pd.DataFrame:
    """Aplica status/reason/updated_at a partir do KYC externo e dos bloqueios internos por risco."""
    users = users.copy()
    kyc_status = kyc["status"].to_numpy()

    users["status"] = np.select([kyc_status == "declined", kyc_status == "blocked"], ["inactive", "blocked"], "active")
    users["reason"] = np.where(kyc_status == "approved", None, kyc["reason"].to_numpy())
    updated = kyc["updated_s"].to_numpy(dtype="int64").copy()

    approved = np.where(kyc_status == "approved")[0]
    changed = approved[rng.random(len(approved)) < CADASTRAL_UPDATE_SHARE]
    updated[changed] = rng.integers(updated[changed], END_S)

    index_of = {uid: k for k, uid in enumerate(users["user_id"])}
    status = users["status"].to_numpy(dtype=object)
    reason = users["reason"].to_numpy(dtype=object)
    for user_id, block_s in risk_block.items():
        k = index_of[user_id]
        status[k] = "blocked"
        reason[k] = "BLOCKED_BY_RISK"
        updated[k] = block_s
    users["status"] = status
    users["reason"] = reason
    users["updated_s"] = updated
    return users


def format_users(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["created_at"] = to_datetime_str(out["created_s"])
    out["updated_at"] = to_datetime_str(out["updated_s"])
    out["income"] = to_money(out["income"])
    out["net_worth"] = to_money(out["net_worth"])
    return out[USER_COLUMNS]
