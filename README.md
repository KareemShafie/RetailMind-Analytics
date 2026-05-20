# 🛒 RetailMind Analytics

**Interactive Data Mining Dashboard for Business Insights + Smart Product Recommendations**

RetailMind Analytics is an interactive data mining dashboard built with **Streamlit** to help retail and e-commerce businesses understand their data in a clear and practical way.

The system transforms transactional retail data into meaningful business insights such as customer groups, product relationships, sales trends, unusual orders, and smart product recommendations.

This project was developed for the **Data Mining Course** at **An-Najah National University**.

---

# 👥 Team Members

- **Abd Alkareem Shafie**
- **Jana Jwabreh**
- **Bushra Hurani**

---

# 🎯 Project Idea

Many retail businesses collect large amounts of sales data but do not always know how to transform it into actionable business decisions.

RetailMind Analytics was designed to bridge the gap between raw retail data and practical business intelligence.

The dashboard helps businesses:

- Understand customer behavior
- Discover valuable customer groups
- Find products frequently bought together
- Monitor sales performance over time
- Detect unusual transactions
- Recommend products to customers
- Generate business-friendly insights and reports

The project was designed with both **technical accuracy** and **business usability** in mind.

---

# ✨ Main Features

## 📊 Data Preprocessing

The project includes a flexible preprocessing pipeline capable of handling multiple retail and e-commerce datasets.

The preprocessing module handles:

- Missing values
- Duplicate removal
- Invalid value filtering
- Column mapping
- Data type correction
- Standard schema preparation
- Feature engineering
- Data quality reporting

The preprocessing stage makes the dashboard adaptable to multiple datasets instead of being limited to a single file format.

---

## 👥 Customer Segmentation & Clustering

The dashboard applies clustering techniques to group customers based on purchasing behavior.

Customer behavior features include:

- Total spending
- Number of orders
- Average order value
- Product diversity
- Customer activity period
- Recency of purchases

Implemented techniques:

- KMeans Clustering
- Hierarchical Clustering
- DBSCAN
- PCA / SVD Visualization
- Clustering Comparison
- Business Interpretation of Clusters

Instead of only showing technical cluster IDs, the dashboard explains customer groups in a business-friendly way to support marketing and business strategies.

---

## 🛍️ Product Relationship Mining

This module uses association rule mining to discover products frequently bought together.

The system helps identify:

- Product bundles
- Cross-selling opportunities
- Frequently co-purchased items
- Marketing campaign opportunities

Implemented concepts:

- Support
- Confidence
- Lift
- Market Basket Analysis
- Association Rules

---

## 📈 Sales Trends Analysis

The Sales Trends module analyzes revenue and customer activity over time.

Features include:

- Daily revenue trends
- Monthly sales overview
- Rolling averages
- Sales heatmaps
- Peak sales period analysis
- Forecasting support

This helps businesses monitor performance and improve planning decisions.

---

## 🚨 Anomaly Detection

The dashboard includes anomaly detection to identify unusual transactions and customer behavior.

Examples include:

- Very high-value orders
- Unusual quantities
- Suspicious customer behavior
- Potential data entry issues
- Orders requiring business review

---

## 🎯 Smart Recommendation System

RetailMind Analytics includes a smart recommendation engine that suggests products based on customer purchase behavior.

Features include:

- Customer-based recommendations
- Basket add-on recommendations
- Popular product fallback recommendations
- Cross-selling opportunities
- Business-friendly recommendation explanations

The recommendation system helps businesses improve customer experience and increase average order value.

---

## 👤 Customer Profile Analysis

The Customer Profile section allows detailed inspection of customer behavior.

The system summarizes:

- Customer spending
- Purchased products
- Order history
- Customer activity
- Suggested business actions

---

# 🖥️ Dashboard Sections

The dashboard is organized into multiple interactive sections:

- Overview
- Data Quality
- Customer Groups
- What Sells Together?
- Sales Trends
- Unusual Orders
- Recommendations
- Customer Profile

The interface was designed to be understandable for both technical and non-technical users.

---

# 📚 Dataset Support

The system supports multiple retail and e-commerce transaction datasets.

### Expected Dataset Fields

The dashboard works best with datasets containing fields similar to:

- Invoice / Order ID
- Product Code
- Product Description
- Quantity
- Unit Price
- Invoice Date
- Customer ID
- Country

The preprocessing pipeline automatically maps columns into a standardized structure.

---

# 🧠 Data Mining Techniques Used

The project applies multiple data mining techniques, including:

- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Clustering
- Association Rule Mining
- Market Basket Analysis
- Time Series Analysis
- Anomaly Detection
- Recommendation Systems

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Main Libraries & Tools

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Mlxtend
- Matplotlib

---

# 📂 Project Structure

```text
RetailMind-Analytics/
│
├── app.py
├── requirements.txt
│
├── modules/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── association_rules.py
│   ├── time_series.py
│   ├── anomaly_detection.py
│   ├── recommender.py
│   └── reporting.py
│
└── data/
```

---

# 🚀 Installation & Running the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/KareemShafie/RetailMind-Analytics.git
```

---

## 2️⃣ Open the Project Folder

```bash
cd RetailMind-Analytics
```

---

## 3️⃣ Install Required Libraries

If requirements.txt exists:

```bash
pip install -r requirements.txt
```

If requirements.txt is missing:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib plotly mlxtend
```

---

## 4️⃣ Run the Dashboard

```bash
streamlit run app.py
```

After running the command, the dashboard will open in the browser at:

```text
http://localhost:8501
```

---

# 👨‍💻 My Contributions

Main contributions by **Abd Alkareem Shafie**:

- Data Preprocessing Pipeline
- Clustering Module
- Recommendation System
- Streamlit Application Integration (`app.py`)
- Business-Oriented Dashboard Improvements

---

# 💼 Business Value

RetailMind Analytics helps businesses:

- Understand customer purchasing behavior
- Improve marketing campaigns
- Discover customer segments
- Detect unusual transactions
- Build product bundles
- Improve recommendation quality
- Support data-driven decision-making

The dashboard combines data mining techniques with practical business intelligence applications.

---

# ⭐ What Makes This Project Different?

Unlike traditional academic implementations, RetailMind Analytics combines multiple data mining tasks inside one complete interactive business dashboard.

### Key strengths include:

- Multi-module analytics system
- Business-friendly interface
- Flexible multi-dataset support
- Smart recommendation engine
- Customer segmentation with interpretation
- Interactive preprocessing quality reports
- Association rule mining
- Time series analysis
- Anomaly detection
- Exportable business insights

---

# 📖 Academic Purpose

This project was created for academic and educational purposes as part of the Data Mining course.

It demonstrates how data mining techniques can be applied to real-world retail and e-commerce datasets to support business intelligence and decision-making.

---

## 📜 License

This project is protected by a custom non-commercial academic license.

The software may be used, copied, and modified for personal, educational, and non-commercial purposes with proper credit to the authors.

Commercial use, selling, paid hosting, or distributing the project for profit requires prior written permission from the authors.
