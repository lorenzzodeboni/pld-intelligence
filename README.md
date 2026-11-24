# PLD Intelligence

Protótipo de apoio ao analista de PLD do desafio Banco do Brasil / FIAP. A base sintética em `data/` já estava pronta. Esta camada lê cadastro, conta, movimentações e KYC externo, aponta tipologias e redige o dossiê. A decisão continua com o analista.

## Rodar

Na pasta do projeto:

```bat
run.bat
```

Ou:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Conferir as regras sem abrir a tela:

```bat
.venv\Scripts\python.exe src\pld\evaluate.py
```

## O que a tela faz

- Fila ordenada por score, com filtro de nível e de indicador.
- Dossiê do cliente: indicadores, extrato, gráfico e relatório para baixar.
- Copiloto para explicar o caso, listar críticos e achar contas externas usadas por vários clientes.
- Registro local da decisão (em análise, comunicar, encerrar, falso positivo).

O score soma pesos de regras explícitas. `BLOCKED_BY_RISK` aparece como contexto e não entra no cálculo. Sem `OPENAI_API_KEY` ou `GROQ_API_KEY`, o texto sai só das regras. Com a chave, o interruptor "Reescrever com LLM" pede uma redação limitada a essas evidências.
