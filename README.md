# Drug Price Optimization

A multi-agent pipeline, built with [Agno](https://github.com/agno-agi/agno), that turns
CMS NADAC (National Average Drug Acquisition Cost) data into pricing strategy
recommendations.

## Pipeline

| # | Agent | Module | Output |
|---|-------|--------|--------|
| 1 | `DataFoundationAgent` | `agents/data_foundation_agent.py` | `outputs/foundation_output.json` |
| 2 | `MarketIntelligenceAgent` | `agents/market_intelligence_agent.py` | `outputs/intelligence_output.json` |
| 3 | `ProfitabilityConstraintAgent` | `agents/profitability_agent.py` | `outputs/profitability_output.json` |
| 4 | `StrategyRecommendationAgent` | `agents/agno_pricing_strategy_agent.py` | `outputs/recommendations_output.json` |

Each agent consumes the previous agent's JSON output. Agent 4 takes both the
intelligence and profitability outputs.

## Project layout

```
.
├── data/raw/                      # Source NADAC dataset (CSV)
├── notebooks/
│   ├── pipeline.ipynb             # Runs the agents end to end
│   └── evaluation.ipynb           # Evaluates the final recommendations
├── outputs/                       # JSON outputs from each pipeline stage
├── src/drug_price_optimization/
│   └── agents/                    # The four pipeline agents
├── pyproject.toml
└── uv.lock
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and installs the package in editable mode,
so notebooks can `from drug_price_optimization.agents... import ...`.

## Usage

Open the notebooks and select the `.venv` kernel, then run
`notebooks/pipeline.ipynb` followed by `notebooks/evaluation.ipynb`.
Notebook paths are relative to the `notebooks/` folder.

Or run the pipeline from the command line:

```bash
uv run python -c "
import json, pandas as pd
from drug_price_optimization.agents.data_foundation_agent import DataFoundationAgent
from drug_price_optimization.agents.market_intelligence_agent import MarketIntelligenceAgent
from drug_price_optimization.agents.profitability_agent import ProfitabilityConstraintAgent
from drug_price_optimization.agents.agno_pricing_strategy_agent import StrategyRecommendationAgent

df = pd.read_csv('data/raw/cleaned_nadac-national-average-drug-acquisition-cost-05-14-2025.csv')
foundation = DataFoundationAgent().process_nadac_data(df)
intel = MarketIntelligenceAgent().analyze_market_intelligence(foundation)
profit = ProfitabilityConstraintAgent().calculate_optimal_pricing(intel)
recs = StrategyRecommendationAgent().process_portfolio(intel, profit)
json.dump(recs, open('outputs/recommendations_output.json', 'w'), indent=2)
"
```

## Notes

- `agno` is pinned to `2.0.x`. Newer releases (2.1+/3.x) reject the
  `model="microsoft/DialoGPT-..."` strings the agents pass to `Agent`.
