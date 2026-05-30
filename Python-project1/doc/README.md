# Analyzing Customer Orders Using Python

Course-end project for processing and analyzing e-commerce customer orders with Python built-in data structures (lists, tuples, dictionaries, sets) and control flow (loops, conditionals).

## Project overview

You act as a data analyst for an e-commerce company selling products in **Electronics**, **Clothing**, and **Home Essentials**. The goals are to:

- Classify customers by total spending (high / moderate / low value)
- Analyze revenue and trends by product category
- Surface insights for marketing and inventory decisions

## Repository layout

```
Python-project1/
├── data/
│   └── customer_orders.json    # Single source dataset (orders, mappings, reference outputs)
├── doc/
│   ├── README.md               # This file
│   ├── PRODUCT_PLAN.md         # Product goals, scope, deliverables
│   ├── TECHNICAL_PLAN.md       # Architecture, environment setup, implementation tasks
│   └── 1745495045_analyzing_customer_orders_using_python.pdf
├── src/
│   └── customer_order_analysis.py  # Single class with all tasks
└── requirements.txt
```

## Dataset

All order data lives in one file:

**`../data/customer_orders.json`**

| Section | Purpose |
|--------|---------|
| `customer_names` | List of customers |
| `orders` | Each order: customer, product, price, category |
| `product_categories` | Product → category mapping |
| `customer_products` | Customer → list of products (dictionary view) |
| `unique_categories` | Electronics, Clothing, Home Essentials |
| `classification_rules` | Spending thresholds for buyer tiers |
| `expected_analysis` | Reference totals for validation (optional when learning) |

Load the JSON in Python:

```python
import json
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "customer_orders.json"
with open(data_path, encoding="utf-8") as f:
    dataset = json.load(f)
```

## Course tasks (from problem statement)

1. **Store customer orders** — lists, tuple-like order records, customer → products dictionary  
2. **Classify products by category** — product→category dict, unique categories set  
3. **Analyze customer orders** — total spend per customer; classify high / moderate / low  
4. **Generate business insights** — category revenue, unique products, electronics buyers, top 3 spenders  
5. **Organize and display** — summaries, multi-category customers, electronics ∩ clothing  

## Buyer classification

| Tier | Rule |
|------|------|
| High-value | Total spending **> $100** |
| Moderate | **$50 – $100** (inclusive) |
| Low-value | **< $50** |

## Deliverables

- Python scripts or notebook performing all five task groups  
- Report covering customer classification, category sales, and business insights  
- Optional: compare your script output to `expected_analysis` in the JSON  

## Related documents

- [PRODUCT_PLAN.md](./PRODUCT_PLAN.md) — product scope, stakeholders, success metrics  
- [TECHNICAL_PLAN.md](./TECHNICAL_PLAN.md) — environment setup, module design, implementation checklist  
- [Course PDF](./1745495045_analyzing_customer_orders_using_python.pdf) — official problem statement  

## Quick start

```bash
cd /path/to/Python-project1
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 src/customer_order_analysis.py
```

See **TECHNICAL_PLAN.md** for full environment and project setup steps.
