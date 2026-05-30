# Technical Plan — Analyzing Customer Orders Using Python

## 1. Architecture overview

```
customer_orders.json
        │
        │ json.load (in __init__)
        ▼
customer_order_analysis.py
  CustomerOrderAnalysis class
        │
        ├── task1_store_orders()
        ├── task2_classify_products()
        ├── task3_analyze_orders()
        ├── task4_business_insights()
        ├── task5_display_data()
        └── run_all()
        │
        ▼
   print output (terminal)
```

**Design principle:** One simple Python file with one class. JSON is the data source; lists, tuples, dictionaries, and sets are used for analysis — matching the course requirements.

## 2. Technology stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.10+ | 3.9+ sufficient |
| Data format | JSON | UTF-8, single file under `data/` |
| Dependencies | stdlib only | `json`, `pathlib` |
| OS | Linux / macOS / Windows | Paths via `pathlib` |

## 3. Environment setup

### 3.1 Prerequisites

- Python 3.10 or newer (`python3 --version`)
- Text editor or IDE

### 3.2 Virtual environment (optional)

```bash
cd Python-project1
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 Verify data file

```bash
python3 -c "from pathlib import Path; print('OK', Path('data/customer_orders.json').exists())"
```

Expected: `OK True`

## 4. Data model mapping (JSON → Python)

| JSON field | Python structure | Course concept |
|------------|------------------|----------------|
| `customer_names` | `list` | List |
| Each `orders[]` item | `(name, product, price, category)` tuple in `list` | Tuple in list |
| `customer_products` | `dict` | Dictionary |
| `product_categories` | `dict` | Dictionary |
| Unique categories | `set` | Set |
| Unique products | `set` | Set |
| Category revenue | `dict` | Dictionary |

## 5. Class methods (single file)

| Method | Task | PDF task |
|--------|------|----------|
| `__init__` | Load JSON, build order tuples | Task 1 |
| `task1_store_orders()` | Print lists, tuples, dictionary | Task 1 |
| `task2_classify_products()` | Product→category dict, categories set | Task 2 |
| `task3_analyze_orders()` | Total spend loop, if/else classification | Task 3 |
| `task4_business_insights()` | Category revenue, unique products, top 3 | Task 4 |
| `task5_display_data()` | Multi-category sets, electronics ∩ clothing | Task 5 |
| `run_all()` | Run all tasks in order | All |

## 6. Implementation checklist

### Task 1 — Store customer orders

- [x] `customer_names` list from JSON
- [x] Order tuples in `self.orders`
- [x] `customer_products` dictionary

### Task 2 — Classify products by category

- [x] `product_categories` dictionary
- [x] Unique categories using `set()`

### Task 3 — Analyze customer orders

- [x] Loop to calculate total spend per customer
- [x] Classify: >$100 high, $50–$100 moderate, <$50 low

### Task 4 — Business insights

- [x] Revenue per category dictionary
- [x] Unique products set
- [x] Electronics customers (list comprehension)
- [x] Top 3 customers (sorting)

### Task 5 — Organize and display

- [x] Print customer spending summary
- [x] Customers with multiple categories (set)
- [x] Electronics AND Clothing intersection

## 7. Execution

```bash
cd Python-project1
python3 src/customer_order_analysis.py
```

The `if __name__ == "__main__":` block at the bottom of the file creates `CustomerOrderAnalysis()` and calls `run_all()`.

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` for JSON | Run from `Python-project1` project root |
| Wrong classification at $50 / $100 | Moderate band is `$50 <= total <= $100` |
| `venv` not active | Run `source .venv/bin/activate` if using virtual env |

## 9. Document index

| Document | Path |
|----------|------|
| Overview & quick start | `doc/README.md` |
| Product scope | `doc/PRODUCT_PLAN.md` |
| This technical plan | `doc/TECHNICAL_PLAN.md` |
| Dataset | `data/customer_orders.json` |
| Main code | `src/customer_order_analysis.py` |
