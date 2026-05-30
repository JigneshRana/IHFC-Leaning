"""
Analyzing Customer Orders Using Python
Learning project - one class with simple functions for each task.
"""

import json
from pathlib import Path


class CustomerOrderAnalysis:
    """Analyze customer orders using lists, tuples, dictionaries, and sets."""

    def __init__(self, data_file="data/customer_orders.json"):
        # Load JSON data
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / data_file

        with open(file_path, encoding="utf-8") as file:
            self.data = json.load(file)

        # Task 1: Store customer orders
        self.customer_names = self.data["customer_names"]
        self.customer_products = self.data["customer_products"]
        self.product_categories = self.data["product_categories"]

        # Convert orders to list of tuples: (customer, product, price, category)
        self.orders = []
        for order in self.data["orders"]:
            order_tuple = (
                order["customer_name"],
                order["product"],
                order["price"],
                order["category"],
            )
            self.orders.append(order_tuple)

    # ----- Task 1: Store customer orders -----
    def task1_store_orders(self):
        print("\n" + "=" * 50)
        print("TASK 1: Store Customer Orders")
        print("=" * 50)

        print("\nCustomer names (list):")
        for name in self.customer_names:
            print(" ", name)

        print("\nSample order tuples:")
        for order in self.orders[:3]:
            print(" ", order)

        print("\nCustomer products (dictionary):")
        for customer, products in self.customer_products.items():
            print(f"  {customer}: {products}")

    # ----- Task 2: Classify products by category -----
    def task2_classify_products(self):
        print("\n" + "=" * 50)
        print("TASK 2: Classify Products by Category")
        print("=" * 50)

        # Set of unique categories
        categories = set(self.product_categories.values())

        print("\nProduct -> Category mapping:")
        for product, category in self.product_categories.items():
            print(f"  {product} -> {category}")

        print("\nAvailable product categories (set):")
        for category in sorted(categories):
            print(" ", category)

        return categories

    # ----- Task 3: Analyze customer orders -----
    def task3_analyze_orders(self):
        print("\n" + "=" * 50)
        print("TASK 3: Analyze Customer Orders")
        print("=" * 50)

        # Calculate total spending for each customer
        customer_spending = {}
        for customer, product, price, category in self.orders:
            if customer not in customer_spending:
                customer_spending[customer] = 0
            customer_spending[customer] = customer_spending[customer] + price

        # Classify customers
        customer_classification = {}
        for customer, total in customer_spending.items():
            if total > 100:
                customer_classification[customer] = "high-value buyer"
            elif total >= 50:
                customer_classification[customer] = "moderate buyer"
            else:
                customer_classification[customer] = "low-value buyer"

        print("\nCustomer spending and classification:")
        for customer in sorted(customer_spending.keys()):
            total = customer_spending[customer]
            label = customer_classification[customer]
            print(f"  {customer}: ${total:.2f} -> {label}")

        self.customer_spending = customer_spending
        self.customer_classification = customer_classification

        return customer_spending, customer_classification

    # ----- Task 4: Generate business insights -----
    def task4_business_insights(self):
        print("\n" + "=" * 50)
        print("TASK 4: Generate Business Insights")
        print("=" * 50)

        # Total revenue per category
        category_revenue = {}
        for customer, product, price, category in self.orders:
            if category not in category_revenue:
                category_revenue[category] = 0
            category_revenue[category] = category_revenue[category] + price

        print("\nRevenue by category:")
        for category, revenue in sorted(category_revenue.items()):
            print(f"  {category}: ${revenue:.2f}")

        # Unique products using set
        unique_products = set()
        for customer, product, price, category in self.orders:
            unique_products.add(product)

        print(f"\nUnique products ordered ({len(unique_products)}):")
        print(" ", sorted(unique_products))

        # Customers who purchased electronics (list comprehension)
        electronics_customers = [
            customer
            for customer, product, price, category in self.orders
            if category == "Electronics"
        ]
        electronics_customers = sorted(set(electronics_customers))

        print("\nCustomers who purchased Electronics:")
        print(" ", electronics_customers)

        # Top 3 highest spending customers
        spending_list = list(self.customer_spending.items())
        spending_list.sort(key=lambda x: x[1], reverse=True)
        top_three = spending_list[:3]

        print("\nTop 3 highest spending customers:")
        for i, (customer, total) in enumerate(top_three, start=1):
            print(f"  {i}. {customer}: ${total:.2f}")

        self.category_revenue = category_revenue
        self.unique_products = unique_products

        return category_revenue, unique_products, top_three

    # ----- Task 5: Organize and display data -----
    def task5_display_data(self):
        print("\n" + "=" * 50)
        print("TASK 5: Organize and Display Data")
        print("=" * 50)

        # Categories purchased by each customer
        customer_categories = {}
        for customer, product, price, category in self.orders:
            if customer not in customer_categories:
                customer_categories[customer] = set()
            customer_categories[customer].add(category)

        # Customers who purchased from multiple categories
        multi_category_customers = []
        for customer, categories in customer_categories.items():
            if len(categories) > 1:
                multi_category_customers.append(customer)

        print("\nCustomers who purchased from multiple categories:")
        print(" ", sorted(multi_category_customers))

        # Customers who bought both electronics and clothing
        electronics_buyers = set()
        clothing_buyers = set()
        for customer, categories in customer_categories.items():
            if "Electronics" in categories:
                electronics_buyers.add(customer)
            if "Clothing" in categories:
                clothing_buyers.add(customer)

        common_customers = electronics_buyers & clothing_buyers

        print("\nCustomers who bought both Electronics and Clothing:")
        print(" ", sorted(common_customers))

        # Summary
        print("\nSummary:")
        print(f"  Total customers: {len(self.customer_names)}")
        print(f"  Total orders: {len(self.orders)}")
        print(f"  Total revenue: ${sum(self.customer_spending.values()):.2f}")

    def run_all(self):
        """Run all tasks one by one."""
        print("Analyzing Customer Orders Using Python")
        self.task1_store_orders()
        self.task2_classify_products()
        self.task3_analyze_orders()
        self.task4_business_insights()
        self.task5_display_data()
        print("\nAnalysis complete.")


if __name__ == "__main__":
    analysis = CustomerOrderAnalysis()
    analysis.run_all()
