# ABM Wyckoff Simulator

Agent-based model (ABM) that simulates the emergence of Wyckoff-style
accumulation/distribution patterns in a synthetic market, driven by the
interaction of a Smart Money agent and several types of retail agents
(trend followers, contrarians, noise traders, algorithmic traders).

This repository contains the simulation backend (FastAPI). It exposes a
single endpoint that runs one simulation and returns OHLC candle data and
Smart Money inventory over time as JSON.

## Requirements

- Python 3.10+
- fastapi
- uvicorn
- numpy
- pydantic

Install with:

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn main_api_ABM:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive OpenAPI
docs at `http://127.0.0.1:8000/docs`.

## Usage

Send a POST request to `/api/v1/simulate` with a JSON body (all fields are
optional; defaults are listed in the `SimulationInput` model):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"seed": 67, "logica_sm": "flujo"}'
```

The response contains:

- `candles`: a list of 200 OHLC candles (`open`, `high`, `low`, `close`),
  each aggregating a chunk of the simulated ticks
- `inventory_sm`: the Smart Money agent's aggregate inventory at the end
  of each of those 200 chunks

### Visualizing the output

This repository only returns raw simulation data as JSON — it does not
include plotting or post-processing code. To visualize the candlestick
chart and the inventory curve, you can either:

- use the interactive web demo: `[URL pending]`, or
- write your own plotting script from the JSON response (e.g. with
  `matplotlib` or `mplfinance`), using `candles` and `inventory_sm` as
  input.

## Model parameters

See `SimulationInput` in `main_api_ABM.py` for the full list of
configurable parameters (agent proportions, price-impact coefficients,
Smart Money logic selector, stop-loss behavior, etc.). A detailed
description of the model and its parameters is provided in the
accompanying paper.

## Smart Money logics

The simulator supports two mutually exclusive Smart Money decision
rules, selected via the `logica_sm` field:

- `"original"`: distance-to-fundamental-value logic (distributes once
  price has moved a fixed percentage away from the accumulation low)
- `"flujo"`: revised logic that triggers distribution based on an
  aggregate market-flow signal instead

## Citation

If you use this code, please cite: `[citation pending publication]`

## License

`[MIT]`
