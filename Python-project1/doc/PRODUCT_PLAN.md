# Product Plan — Analyzing Customer Orders Using Python

## 1. Executive summary

This course-end project simulates real e-commerce analytics: turning raw customer orders into classifications and category-level insights that support marketing and inventory decisions. The product outcome is an **analyst-ready dataset** plus a **repeatable analysis report**, not a production web application.

## 2. Problem statement

An e-commerce company sells across **Electronics**, **Clothing**, and **Home Essentials**. Leadership needs answers to:

- Who are our highest-value customers?
- Which categories drive the most revenue?
- Which customers buy across multiple categories (cross-sell opportunity)?
- Who buys both electronics and clothing (bundle / campaign targets)?

**Situation:** You are the data analyst. **Constraint:** Use Python core data structures only (no database required for the learning exercise).

## 3. Goals and objectives

| Goal | Objective | Success indicator |
|------|-----------|-------------------|
| Customer segmentation | Classify every customer by lifetime order value in dataset | 100% customers labeled high / moderate / low |
| Category performance | Revenue by category | Dictionary of category → total USD |
| Product catalog | Unique products and categories | Sets derived from orders |
| Actionable lists | Top 3 spenders, electronics buyers, cross-category buyers | Printed or exported lists match manual checks |
| Decision support | Written insights for managers | Report with 3+ concrete recommendations |

## 4. Scope

### In scope

- Storing and loading order data (JSON → Python structures)
- Per-customer total spend and tier classification
- Category revenue aggregation
- Set-based insights (unique products, multi-category customers, electronics ∩ clothing)
- Sorted ranking (top 3 customers by spend)
- Summary report for submission

### Out of scope (for this learning project)

- Live API / database integration
- Real-time dashboards
- Machine learning forecasting
- Payment / fraud analysis
- User authentication or multi-tenant systems

## 5. Stakeholders and users

| Role | Need |
|------|------|
| Business manager | Category revenue, top customers, segment counts |
| Marketing | Lists for campaigns (electronics buyers, cross-category) |
| Inventory / ops | Popular categories and product variety |
| Learner (you) | Demonstrate lists, tuples, dicts, sets, loops, conditionals |

## 6. Data product

**Primary artifact:** `data/customer_orders.json`

- 10 customers, 100 order line items
- 3 categories (Electronics, Clothing, Home Essentials), 30 products in catalog
- Embedded `classification_rules` and `expected_analysis` for self-checking

**Secondary artifact:** Analysis report (PDF/Markdown) submitted per course instructions.

## 7. Functional requirements (mapped to course tasks)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Maintain list of customer names | P0 |
| FR-2 | Store orders as records (name, product, price, category) | P0 |
| FR-3 | Map customer → products ordered | P0 |
| FR-4 | Map product → category; expose unique categories | P0 |
| FR-5 | Compute total spend per customer | P0 |
| FR-6 | Classify customers: >$100 high, $50–$100 moderate, <$50 low | P0 |
| FR-7 | Revenue per category | P0 |
| FR-8 | Unique products across all orders (set) | P1 |
| FR-9 | Customers who bought electronics (list comprehension) | P1 |
| FR-10 | Top 3 customers by spend (sorting) | P1 |
| FR-11 | Customers purchasing from multiple categories (set ops) | P1 |
| FR-12 | Customers who bought electronics AND clothing | P1 |
| FR-13 | Print/display customer summary with classification | P0 |

## 8. Business insights to deliver (report section)

Example themes your report should address using computed numbers:

1. **Category mix** — Which category generates the most revenue? Should inventory favor Electronics?
2. **Customer concentration** — Do a few high-value buyers account for most revenue?
3. **Cross-sell** — Which customers already span categories (target for loyalty programs)?
4. **Segmentation** — How many high vs moderate vs low buyers? Implications for discount strategy.
5. **Product diversity** — Count of unique SKUs; gaps in Home Essentials vs Electronics.

Use figures from your script run; reference `expected_analysis` in JSON only to verify logic while learning.

## 9. Deliverables and timeline (suggested)

| Phase | Deliverable | Duration (guide) |
|-------|-------------|------------------|
| 1 | Environment setup, load JSON | 0.5 day |
| 2 | Tasks 1–2: structures + categories | 0.5 day |
| 3 | Tasks 3–4: spend, classify, insights | 1 day |
| 4 | Task 5: display + set operations | 0.5 day |
| 5 | Written report + submission | 1 day |

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Boundary errors ($50, $100) | Document inclusive moderate band; add unit tests for edge cases |
| Duplicate product names | Use consistent product strings from JSON |
| Hard-coded paths | Use `pathlib` relative to project root |
| Copy-paste without understanding | Implement each task in isolation before combining |

## 11. Success criteria (project complete when)

- [ ] All five task groups from the PDF are implemented in Python
- [ ] Report includes customer classification table
- [ ] Report includes total sales per category
- [ ] Report includes at least three business insights with supporting numbers
- [ ] Code runs in a documented virtual environment without manual data edits

## 12. References

- Course PDF: `doc/1745495045_analyzing_customer_orders_using_python.pdf`
- Dataset: `data/customer_orders.json`
- Technical implementation: `doc/TECHNICAL_PLAN.md`
