# Analyzing Customer Orders Using Python

Simple learning project to analyze e-commerce orders using Python data structures.

## Run

```bash
cd Python-project1
python3 src/customer_order_analysis.py
```

## Project structure

```
Python-project1/
├── data/customer_orders.json       # Dataset (10 customers, 100 orders)
├── doc/                            # Project documentation
└── src/
    └── customer_order_analysis.py  # Single class with all tasks
```

## Tasks (in customer_order_analysis.py)

| Method | Task |
|--------|------|
| `task1_store_orders()` | Lists, tuples, dictionaries |
| `task2_classify_products()` | Product categories and sets |
| `task3_analyze_orders()` | Customer spending and classification |
| `task4_business_insights()` | Revenue, unique products, top 3 customers |
| `task5_display_data()` | Set operations and summary |
| `run_all()` | Runs all tasks |

## Buyer classification

- **High-value:** total spending > $100
- **Moderate:** $50 to $100
- **Low-value:** below $50
