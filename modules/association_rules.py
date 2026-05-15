"""
Module 3: Market Basket Analysis — Association Rule Mining (Apriori)
MARKETING-FRIENDLY VERSION with Business Language & Advanced Metrics
Responsible Student: Jana jawabreh

Technique: Apriori Algorithm
- Finds frequent itemsets using minimum support threshold
- Generates association rules using minimum confidence threshold
- Calculates lift to identify real (non-coincidental) relationships
- Translates technical metrics into business-friendly language
"""

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

import plotly.graph_objects as go
import streamlit as st


@st.cache_data(show_spinner=False)
def build_basket_matrix(
    df: pd.DataFrame,
    country: str = 'United Kingdom',
    max_products: int = 80,
) -> pd.DataFrame:
    work = df.copy()

    if country and country != 'All':
        work = work[work['Country'] == country]

    work = work[~work['InvoiceNo'].astype(str).str.startswith('C')]
    work = work[work['Quantity'] > 0]
    work = work.dropna(subset=['Description'])

    invoice_sizes = work.groupby('InvoiceNo').size()
    valid_invoices = invoice_sizes[invoice_sizes <= 30].index
    work = work[work['InvoiceNo'].isin(valid_invoices)]

    if work.empty:
        return pd.DataFrame()

    top_products = (
        work.groupby('Description')['Quantity']
        .sum()
        .nlargest(max_products)
        .index
    )
    work = work[work['Description'].isin(top_products)]

    basket = (
        work.groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum()
        .unstack(fill_value=0)
    )

    return basket.gt(0).astype(bool)


@st.cache_data(show_spinner=False)
def generate_association_rules(
    basket: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.3,
) -> pd.DataFrame:
    if basket.empty:
        return pd.DataFrame()

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        low_memory=True,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_confidence,
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules['lift'] > 1].copy()

    if rules.empty:
        return pd.DataFrame()

    # Convert frozensets to readable strings
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: ', '.join(sorted(list(x)))
    )
    rules['consequents'] = rules['consequents'].apply(
        lambda x: ', '.join(sorted(list(x)))
    )

    for col in ['support', 'confidence', 'lift', 'antecedent support', 'consequent support']:
        if col in rules.columns:
            rules[col] = rules[col].round(4)

    return rules.sort_values(
        ['lift', 'confidence'], ascending=False
    ).reset_index(drop=True)


def classify_rule_strength(lift: float) -> dict:
    if lift > 8:
        return {
            'strength_level': '🔥 HOT DEAL',
            'strength_name': 'Extremely Strong Connection',
            'business_action': '⚡ Create bundle immediately - offer as combo deal',
            'color': '#FF6B6B',
            'priority': 1,
            'marketing_message': 'These products have an extremely strong connection. Customers who buy one are significantly more likely to buy the other.'
        }
    elif lift > 4:
        return {
            'strength_level': '💎 PREMIUM OPPORTUNITY',
            'strength_name': 'Strong Connection',
            'business_action': '📦 Bundle these products - excellent cross-sell potential',
            'color': '#FFA726',
            'priority': 2,
            'marketing_message': 'Strong relationship detected. Excellent for bundling and promotional offers.'
        }
    elif lift > 2:
        return {
            'strength_level': '✅ GOOD OPPORTUNITY',
            'strength_name': 'Moderate Connection',
            'business_action': '💡 Recommend in email campaigns or product pages',
            'color': '#66BB6A',
            'priority': 3,
            'marketing_message': 'Moderate connection found. Worth promoting together in campaigns.'
        }
    elif lift >= 1:
        return {
            'strength_level': '📌 MONITOR',
            'strength_name': 'Weak Connection',
            'business_action': '👀 Track performance - may develop into stronger opportunity',
            'color': '#FDD835',
            'priority': 4,
            'marketing_message': 'Weak but positive connection. Monitor for potential growth.'
        }
    else:
        return {
            'strength_level': '❌ LOW VALUE',
            'strength_name': 'Not Recommended',
            'business_action': '⏭️ Skip for now - focus on stronger patterns',
            'color': '#BDBDBD',
            'priority': 5,
            'marketing_message': 'This relationship is too weak for marketing action.'
        }


def explain_metrics_simple() -> str:
    return """
**🎯 What Do These Numbers Mean?**

**How Common (Support):** How often do customers buy BOTH items together?
- 0.05 = 5% of all purchases include both items
- Higher = more customers are already buying them together

**When Bought (Confidence):** When a customer buys Product A, how often do they also buy Product B?
- 0.60 = 60% of customers who buy A also buy B
- Higher = stronger predictability

**Relationship Strength (Lift):** How much more likely is B when A is bought, compared to random chance?
- 1.0 = No relationship (buying A doesn't help predict B)
- 3.0 = Customers are 3× more likely to buy B when they buy A
- Higher = stronger, more profitable relationship
"""


def get_rules_summary_marketing(rules: pd.DataFrame) -> dict:
    if rules.empty:
        return {}

    top_rule = rules.iloc[0]
    strength_info = classify_rule_strength(top_rule['lift'])

    return {
        'total_rules': len(rules),
        'hot_deals': len(rules[rules['lift'] > 8]),
        'premium_opportunities': len(rules[rules['lift'] > 4]),
        'good_opportunities': len(rules[rules['lift'] > 2]),
        'avg_lift': round(rules['lift'].mean(), 2),
        'max_lift': round(rules['lift'].max(), 2),
        'strongest_rule_antecedent': top_rule['antecedents'],
        'strongest_rule_consequent': top_rule['consequents'],
        'strongest_rule_strength': strength_info['strength_level'],
        'strongest_rule_action': strength_info['business_action'],
    }


def interpret_rules_marketing(rules: pd.DataFrame, top_n: int = 5) -> list:
    if rules is None or rules.empty:
        return ["❌ No strong rules found. Try lowering the thresholds to discover more patterns."]

    insights = []
    for _, row in rules.head(top_n).iterrows():
        strength_info = classify_rule_strength(row['lift'])

        insights.append({
            'rule': f"{row['antecedents']} → {row['consequents']}",
            'strength': strength_info['strength_level'],
            'message': strength_info['marketing_message'],
            'action': strength_info['business_action'],
            'lift': f"{row['lift']:.1f}x",
            'support': f"{row['support']*100:.1f}%",
            'confidence': f"{row['confidence']*100:.1f}%",
        })

    return insights


def plot_rules_marketing_strength(rules: pd.DataFrame):
    if rules.empty:
        return None

    rules_display = rules.head(80).copy()
    rules_display['Strength Category'] = rules_display.apply(
        lambda row: classify_rule_strength(row['lift'])['strength_name'],
        axis=1
    )
    rules_display['Rule Display'] = rules_display['antecedents'] + ' → ' + rules_display['consequents']

    fig = go.Figure()

    colors = {
        'Extremely Strong Connection': '#FF6B6B',
        'Strong Connection': '#FFA726',
        'Moderate Connection': '#66BB6A',
        'Weak Connection': '#FDD835',
        'Not Recommended': '#BDBDBD',
    }

    for strength in ['Extremely Strong Connection', 'Strong Connection', 'Moderate Connection', 'Weak Connection', 'Not Recommended']:
        subset = rules_display[rules_display['Strength Category'] == strength]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['support'],
                y=subset['lift'],
                mode='markers',
                name=strength,
                marker=dict(
                    size=12,
                    color=colors[strength],
                    opacity=0.7,
                    line=dict(width=1, color='white'),
                ),
                text=subset.apply(
                    lambda row: f"<b>{row['Rule Display']}</b><br>" +
                               f"Lift: {row['lift']:.2f}x<br>" +
                               f"Support: {row['support']*100:.1f}%<br>" +
                               f"Confidence: {row['confidence']*100:.1f}%",
                    axis=1
                ),
                hovertemplate='%{text}<extra></extra>',
            ))

    fig.update_layout(
        title='<b>Product Relationships for Marketing Decisions</b><br><sub>X: Market Reach | Y: Profit Potential</sub>',
        xaxis_title='<b>How Common (% of customers buying both)</b>',
        yaxis_title='<b>Business Strength (Lift)</b>',
        template='plotly_white',
        height=600,
        hovermode='closest',
        showlegend=True,
        font=dict(family="Segoe UI, sans-serif", size=11),
    )

    fig.add_hline(y=2, line_dash="dash", line_color="rgba(0,0,0,0.2)",
                  annotation_text="Moderate Opportunity Threshold",
                  annotation_position="right")
    fig.add_hline(y=4, line_dash="dash", line_color="rgba(0,0,0,0.3)",
                  annotation_text="Strong Opportunity Threshold",
                  annotation_position="right")

    return fig


def plot_top_rules_marketing(rules: pd.DataFrame, top_n: int = 15):
    if rules.empty:
        return None

    top = rules.head(top_n).copy()
    top['rule'] = top['antecedents'] + ' → ' + top['consequents']
    top['Strength'] = top.apply(
        lambda row: classify_rule_strength(row['lift'])['strength_name'],
        axis=1
    )

    color_map = {
        'Extremely Strong Connection': '#FF6B6B',
        'Strong Connection': '#FFA726',
        'Moderate Connection': '#66BB6A',
        'Weak Connection': '#FDD835',
        'Not Recommended': '#BDBDBD',
    }
    top['Color'] = top['Strength'].map(color_map)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top.sort_values('lift')['rule'],
        x=top.sort_values('lift')['lift'],
        orientation='h',
        marker=dict(
            color=top.sort_values('lift')['Color'],
            line=dict(color='white', width=2),
        ),
        text=top.sort_values('lift').apply(
            lambda row: f"Lift: {row['lift']:.2f}x",
            axis=1
        ),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
    ))

    fig.update_layout(
        title=f'<b>Top {top_n} Marketing Opportunities</b><br><sub>Ranked by Relationship Strength</sub>',
        xaxis_title='<b>Relationship Strength (Lift)</b>',
        yaxis_title='',
        template='plotly_white',
        height=600,
        font=dict(family="Segoe UI, sans-serif", size=11),
        showlegend=False,
        margin=dict(l=350, r=100),
    )

    fig.update_yaxes(showgrid=False)

    return fig


def create_marketing_report(rules: pd.DataFrame) -> str:
    if rules.empty:
        return "No rules found. Try adjusting the analysis parameters."

    summary = get_rules_summary_marketing(rules)
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    MARKET BASKET ANALYSIS REPORT                           ║
║                        For Marketing Decisions                             ║
╚════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────

✅ Total Opportunities Found: {summary['total_rules']}

🔥 Hot Deals (Immediate action): {summary['hot_deals']}
💎 Premium Opportunities (Strong): {summary['premium_opportunities']}
✅ Good Opportunities (Worth exploring): {summary['good_opportunities']}

Average Relationship Strength: {summary['avg_lift']:.2f}x


TOP PRIORITY RULE
─────────────────────────────────────────────────────────────────────────────
IF customers buy:  {summary['strongest_rule_antecedent']}
THEN they also buy: {summary['strongest_rule_consequent']}

Strength Level: {summary['strongest_rule_strength']}
Recommended Action: {summary['strongest_rule_action']}
Maximum Relationship Strength: {summary['max_lift']:.2f}x


MARKETING RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────

1. 🎯 CREATE BUNDLES
   Focus on "Hot Deal" items - these are proven combinations that customers
   naturally buy together. Create promotional bundles or combo deals.

2. 📧 EMAIL CAMPAIGNS
   Use "Premium Opportunities" for personalized email recommendations.
   When you email about one product, recommend the other.

3. 🛒 PRODUCT PAGE OPTIMIZATION
   Add "customers also buy" sections featuring these related products.

4. 💰 PRICING STRATEGY
   Consider offering small discounts on the second item to encourage
   customers to complete the bundle.

5. 📦 INVENTORY MANAGEMENT
   Stock these paired items near each other or ensure both are available
   to avoid missed sales opportunities.

6. 👥 CUSTOMER SEGMENTATION
   Different customer groups may have different product affinities.
   Consider marketing different bundles to different segments.

═════════════════════════════════════════════════════════════════════════════
"""
    return report