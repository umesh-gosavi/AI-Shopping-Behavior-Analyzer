import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=== AI E-COMMERCE CUSTOMER ANALYTICS PLATFORM ===\n")

# --- STEP 1: LOAD CSV DATA ---
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
transactions = pd.read_csv("transactions.csv")

print(f"Customers: {len(customers)}")
print(f"Products: {len(products)}")
print(f"Transactions: {len(transactions)}")

# --- STEP 2: DATABASE SETUP ---
conn = sqlite3.connect("ecommerce.db")
cursor = conn.cursor()

# Store CSV data in database
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
transactions.to_sql("transactions", conn, if_exists="replace", index=False)

print("\n✅ Data stored in SQLite Database!")

# --- STEP 3: SQL ANALYSIS ---
print("\n=== DATA ANALYSIS ===")

# Total Revenue
cursor.execute("""
    SELECT SUM(p.price * t.quantity) 
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
""")
total_revenue = cursor.fetchone()[0]
print(f"💰 Total Revenue: Rs. {total_revenue:,}")

# Category wise sales
cursor.execute("""
    SELECT p.category, 
           COUNT(*) as total_orders,
           SUM(p.price * t.quantity) as revenue
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY p.category
    ORDER BY revenue DESC
""")
category_sales = cursor.fetchall()
print("\n📊 Category-wise Revenue:")
for row in category_sales:
    print(f"   {row[0]}: {row[1]} orders | Rs. {row[2]:,}")

# Top customers
cursor.execute("""
    SELECT c.name,
           COUNT(*) as total_orders,
           SUM(p.price * t.quantity) as total_spent
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    JOIN products p ON t.product_id = p.product_id
    GROUP BY c.name
    ORDER BY total_spent DESC
    LIMIT 5
""")
top_customers = cursor.fetchall()
print("\n👑 Top 5 Customers:")
for row in top_customers:
    print(f"   {row[0]}: {row[1]} orders | Rs. {row[2]:,}")

# Monthly revenue
cursor.execute("""
    SELECT strftime('%Y-%m', date) as month,
           SUM(p.price * t.quantity) as revenue
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY month
    ORDER BY month
""")
monthly_revenue = cursor.fetchall()
print("\n📅 Monthly Revenue:")
for row in monthly_revenue:
    print(f"   {row[0]}: Rs. {row[1]:,}")

# City wise customers
cursor.execute("""
    SELECT city, COUNT(*) as customers
    FROM customers
    GROUP BY city
    ORDER BY customers DESC
""")
city_data = cursor.fetchall()
print("\n🏙️ City-wise Customers:")
for row in city_data:
    print(f"   {row[0]}: {row[1]} customers")

# --- STEP 4: AI MODEL ---
print("\n=== AI PREDICTION MODEL ===")

# Prepare data
df = pd.merge(transactions, products, on="product_id")
df = pd.merge(df, customers, on="customer_id")

# High value customer — spent more than average
df['total_spent'] = df['price'] * df['quantity']
avg_spent = df.groupby('customer_id')['total_spent'].sum().mean()
df['is_high_value'] = df.groupby('customer_id')['total_spent'].transform('sum').apply(
    lambda x: 1 if x > avg_spent else 0
)

# Encode categorical data
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
df['gender_encoded'] = le.fit_transform(df['gender'])
df['category_encoded'] = le.fit_transform(df['category'])

# Features for AI
X = df[['age', 'city_encoded', 'gender_encoded', 'price', 'quantity', 'rating', 'category_encoded']]
y = df['is_high_value']

# Train AI model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🤖 AI Model Accuracy: {accuracy*100:.1f}%")
print(f"📊 Average Customer Spend: Rs. {avg_spent:,.0f}")

high_value = df[df['is_high_value']==1]['name'].unique()
print(f"👑 High Value Customers: {list(high_value)}")

# --- STEP 5: VISUALIZATIONS ---
print("\n✅ Generating Charts...")

# Chart 1 - Category Revenue
categories = [row[0] for row in category_sales]
revenues = [row[2] for row in category_sales]

plt.figure(figsize=(8,6))
plt.bar(categories, revenues, color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0'])
plt.title("Category-wise Revenue")
plt.xlabel("Category")
plt.ylabel("Revenue (Rs.)")
plt.tight_layout()
plt.savefig("category_revenue.png")
plt.show()

# Chart 2 - Monthly Revenue
months = [row[0] for row in monthly_revenue]
month_rev = [row[1] for row in monthly_revenue]

plt.figure(figsize=(10,6))
plt.plot(months, month_rev, marker='o', color='#2196F3', linewidth=2)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_revenue.png")
plt.show()

# Chart 3 - City wise customers
cities = [row[0] for row in city_data]
city_counts = [row[1] for row in city_data]

plt.figure(figsize=(8,6))
plt.pie(city_counts, labels=cities, autopct='%1.1f%%', startangle=140)
plt.title("City-wise Customer Distribution")
plt.savefig("city_distribution.png")
plt.show()

conn.close()
print("\n🎉 Analysis Complete! Charts saved!")