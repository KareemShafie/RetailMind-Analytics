# 🛒 RetailMind Analytics

**Interactive Data Mining Dashboard for Business Insights + Smart Product Recommendations**

RetailMind Analytics is an interactive data mining dashboard built with **Streamlit** to help retail and e-commerce businesses understand their data in a clear and practical way.

The system takes transactional retail data and turns it into useful business insights such as customer groups, product relationships, sales trends, unusual orders, and smart product recommendations.

This project was developed for the **Data Mining course** at **An-Najah National University**.

---

## 👥 Team Members

- **Abd Alkareem Shafie**
- **Jana Jwabreh**
- **Bushra Hurani**

---

## 🎯 Project Idea

Many retail businesses collect sales data but do not always know how to use it for decision-making.

This project transforms raw transaction data into a business-friendly dashboard that supports:

- Understanding customer behavior
- Finding valuable customer groups
- Discovering products that are often bought together
- Monitoring sales performance
- Detecting unusual orders
- Recommending products to customers
- Generating simple business reports

The main goal is not only to apply data mining algorithms, but also to make the results understandable for business users who may not have a technical background.

---

##  Main Features

### 1. 📊 Data Preprocessing

The system includes a flexible preprocessing pipeline that prepares different retail/e-commerce datasets before analysis.

It handles:

- Missing value handling
- Duplicate removal
- Invalid value filtering
- Column mapping
- Data type correction
- Standard schema preparation
- Feature preparation for later analysis

The preprocessing step helps make the dashboard work with multiple datasets instead of being fixed to only one file.

---

### 2. 👥 Customer Groups

The Customer Groups section applies clustering techniques to group customers based on purchasing behavior.

The system uses customer behavior features such as:

- Total spending
- Number of orders
- Average order value
- Product diversity
- Customer activity period
- Days since last purchase

Instead of only showing technical cluster numbers, the dashboard explains the groups in a business-friendly way so the user can understand who to reward, who to target, and who may need attention.

Used techniques include:

- KMeans Clustering
- Hierarchical Clustering
- DBSCAN
- Clustering comparison
- PCA / SVD visualization
- Business interpretation of segments

---

### 3. 🛍️ Product Relationships

This section uses association rule mining to discover products that customers often buy together.

It helps answer:

- Which products are usually bought together?
- What products can be bundled?
- What items can be used for cross-selling?
- Which product combinations are useful for marketing campaigns?

Used concepts include:

- Support
- Confidence
- Lift
- Market Basket Analysis
- Association Rules

---

### 4. 📈 Sales Trends

The Sales Trends section analyzes revenue over time.

It includes:

- Daily revenue trends
- Monthly revenue overview
- Rolling averages
- Sales heatmaps
- Peak sales periods
- Simple forecasting support

This helps management understand when sales increase or decrease and supports better planning for campaigns and inventory.

---

### 5. 🚨 Unusual Orders

The dashboard includes anomaly detection to find unusual or extreme orders.

This can help identify:

- Very high-value transactions
- Unusual quantities
- Unexpected customer behavior
- Possible data entry problems
- Orders that may need business review

---

### 6. 🎯 Smart Product Recommendations

RetailMind includes a smart recommendation system that suggests products based on customer purchase behavior.

It supports:

- Customer-based recommendations
- Basket add-on recommendations
- Popular product fallback recommendations
- Cross-selling opportunities
- Business-friendly recommendation explanations

The recommender helps businesses suggest products customers are likely to buy next and can support increasing average order value.

---

### 7. 👤 Customer Profile

The Customer Profile section allows the user to inspect customer-level behavior.

It summarizes:

- Customer spending
- Purchased products
- Order history
- Customer activity
- Suggested business actions

---


## 🖥️ Dashboard Tabs

The project is organized into clear dashboard sections:

- Overview
- Data Quality
- Customer Groups
- What Sells Together?
- Sales Trends
- Unusual Orders
- Recommendations
- Customer Profile
- Business Report

The interface was designed to be business-friendly, so the user does not need to understand all data mining algorithms to benefit from the system.

---

##  Dataset Support

The project supports retail and e-commerce transaction datasets that include basic sales information.

### Main expected fields

The system works best with datasets that contain meanings similar to:

- Invoice / Order ID
- Product code or product name
- Product description
- Quantity
- Unit price
- Invoice date
- Customer ID
- Country

The preprocessing module maps columns into a standard format so different datasets can be analyzed using the same dashboard.

---

## 📚 Datasets Used

The project was tested on multiple retail/e-commerce datasets to show that the system is flexible and not hardcoded for only one dataset.

Examples of supported dataset types include:

- Online Retail Dataset
- E-Commerce Transactions Dataset
- Retail Sales Dataset
- Customer Shopping Dataset
- Superstore Sales Dataset

Dataset sources include public retail and e-commerce datasets from:

- UCI Machine Learning Repository
- Kaggle

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Main Tools and Libraries

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Mlxtend
- Matplotlib

---

## 🧠 Data Mining Techniques Used

The project applies several data mining techniques, including:

- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Clustering
- Association Rule Mining
- Market Basket Analysis
- Time Series Analysis
- Anomaly Detection
- Recommendation Systems
- Business Intelligence Reporting

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/janajawabreh267/Data_Mining_Project.git
```

### 2. Open the project folder

```bash
cd RetailMind_Analytics_Final
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

After running the command, the dashboard will open in the browser, usually at:

```text
http://localhost:8501
```

---

## 📌 Project Structure

```text
RetailMind_Analytics_Final/
│
├── app.py
├── requirements.txt
├── LICENSE
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

## 💼 Business Value

RetailMind Analytics can help a retail business:

- Understand sales performance
- Identify important customer groups
- Improve marketing campaigns
- Create product bundles
- Recommend products to customers
- Detect unusual transactions
- Support data-driven decision-making

The dashboard connects data mining techniques with real business use cases.

---

## ⭐ What Makes This Project Different?

This project is not only a technical implementation of algorithms. It combines multiple data mining tasks inside one interactive dashboard and explains the results in a way that business users can understand.

Key strengths:

- Multiple analysis modules in one system
- Business-friendly interface
- Flexible dataset support
- Clear visualizations
- Customer segmentation with actions
- Recommendation system
- Association rules for product bundling
- Sales trend analysis
- Anomaly detection
- Exportable business insights

---

## 📖 Academic Purpose

This project was created for academic purposes as part of the Data Mining course. It demonstrates how data mining techniques can be applied to real-world retail and e-commerce data.

---

## 📜 License

This project is protected by a custom non-commercial academic license.

The software may be used, copied, and modified for personal, educational, and non-commercial purposes with proper credit to the authors.

Commercial use, selling, paid hosting, or distributing the project for profit requires prior written permission from the authors.
