import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7c3aed;
    margin-bottom: 0;
}
.sub-title {
    font-size: 1rem;
    color: #8b8aad;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: #f5f3ff;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #7c3aed;
    margin: 0;
}
.kpi-label {
    font-size: 0.8rem;
    color: #8b8aad;
    margin: 0;
}
.result-box {
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    text-align: center;
}
.result-positive { background: #d1fae5; border: 2px solid #10b981; }
.result-negative { background: #fee2e2; border: 2px solid #ef4444; }
.result-neutral  { background: #fef3c7; border: 2px solid #f59e0b; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.category-tag {
    display: inline-block;
    background: #ede9fe;
    color: #7c3aed;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.3rem;
}
.insight-card {
    background: #faf9ff;
    border-left: 3px solid #7c3aed;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
}
.insight-card.amber { border-left-color: #f59e0b; }
.insight-card.green { border-left-color: #10b981; }
.insight-card.red   { border-left-color: #ef4444; }
.insight-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8b8aad;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('best_sentiment_model.pkl', 'rb') as f:
        sent = pickle.load(f)
    with open('best_category_model.pkl', 'rb') as f:
        cat = pickle.load(f)
    return sent, cat

# ── Text cleaning ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_nlp_tools():
    sw = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    return sw, lem

def clean_text(text, sw, lem):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lem.lemmatize(t) for t in text.split()
              if t not in sw and len(t) > 2]
    return ' '.join(tokens)

def predict(text, sent_bundle, cat_bundle, sw, lem):
    cleaned = clean_text(text, sw, lem)
    sentiment = sent_bundle['label_encoder'].inverse_transform(
        sent_bundle['classifier'].predict(
            sent_bundle['vectorizer'].transform([cleaned])))[0]
    category = cat_bundle['label_encoder'].inverse_transform(
        cat_bundle['classifier'].predict(
            cat_bundle['vectorizer'].transform([cleaned])))[0]
    return sentiment, category

# ── Precomputed stats (from our Kaggle training run) ─────────────────────────
STATS = {
    'total_reviews': 22641,
    'avg_rating': 4.18,
    'pct_positive': 77.1,
    'pct_recommended': 81.9,
    'category_counts': {
        'Sizing & Fit': 18091, 'Quality & Material': 2701,
        'General Feedback': 807, 'Style & Design': 738,
        'Delivery & Shipping': 162, 'Pricing & Value': 128,
    },
    'sentiment_counts': {'Positive': 17448, 'Neutral': 2823, 'Negative': 2370},
    'rating_counts': {1: 821, 2: 1549, 3: 2823, 4: 4908, 5: 12540},
    'dept_avg_rating': {
        'Bottoms': 4.28, 'Intimate': 4.27, 'Jackets': 4.25,
        'Tops': 4.16, 'Dresses': 4.14, 'Trend': 3.84
    },
    'model_results': {
        'sentiment': {
            'Naive Bayes':         {'accuracy': 0.7847, 'f1': 0.7083},
            'Logistic Regression': {'accuracy': 0.8229, 'f1': 0.7947},
            'Linear SVM':          {'accuracy': 0.8167, 'f1': 0.8004},
            'Random Forest':       {'accuracy': 0.7931, 'f1': 0.7262},
            'XGBoost':             {'accuracy': 0.8132, 'f1': 0.7828},
            'LightGBM':            {'accuracy': 0.8196, 'f1': 0.7946},
            'DistilBERT':          {'accuracy': 0.8510, 'f1': 0.8390},
        },
        'category': {
            'Naive Bayes':         {'accuracy': 0.8000, 'f1': 0.7116},
            'Logistic Regression': {'accuracy': 0.8789, 'f1': 0.8572},
            'Linear SVM':          {'accuracy': 0.9185, 'f1': 0.9121},
            'Random Forest':       {'accuracy': 0.8738, 'f1': 0.8484},
            'XGBoost':             {'accuracy': 0.9642, 'f1': 0.9642},
            'LightGBM':            {'accuracy': 0.9702, 'f1': 0.9699},
        }
    }
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["Classifier", "Dashboard", "Model Comparison", "Insights"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#8b8aad; line-height:1.7;">
    Capstone project comparing 7 NLP & ML models for customer feedback classification.<br><br>
    <b>Dataset:</b> Women's E-Commerce Clothing Reviews (22,641 reviews)<br><br>
    <b>Tasks:</b> Sentiment · Complaint Category
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 Customer Feedback Analyser</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Comparative NLP & ML Analysis — Capstone Project</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
if page == "Classifier":
    st.markdown('<p class="section-label">Live Prediction</p>', unsafe_allow_html=True)
    st.markdown("### Classify a Customer Review")
    st.markdown("Paste any product review below. The model will predict its sentiment and complaint category.")

    examples = [
        "The sizing ran really small, had to return it and order two sizes up.",
        "Absolutely love this dress! The fabric is amazing and it fits perfectly.",
        "Very overpriced for what you get. The stitching came apart after one wash.",
        "Delivery took three weeks and the package arrived damaged.",
        "It is okay, nothing special. Not sure I would buy it again.",
        "Beautiful design, very stylish and exactly what I was looking for!",
    ]

    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex[:40] + "…", key=f"ex_{i}"):
            st.session_state['review_text'] = ex

    review = st.text_area(
        "Your review",
        value=st.session_state.get('review_text', ''),
        height=130,
        placeholder="Type or paste a customer review here…"
    )

    if st.button("▶ Classify Review", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                sent_bundle, cat_bundle = load_models()
                sw, lem = get_nlp_tools()
                with st.spinner("Classifying…"):
                    sentiment, category = predict(review, sent_bundle, cat_bundle, sw, lem)

                icons = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}
                css_class = f"result-{sentiment.lower()}"

                st.markdown(f"""
                <div class="result-box {css_class}">
                    <div style="font-size:2.5rem;">{icons.get(sentiment, '🔍')}</div>
                    <p class="result-label">{sentiment}</p>
                    <div class="category-tag">📁 {category}</div>
                    <p style="font-size:0.78rem; color:#6b7280; margin-top:0.75rem;">
                        Sentiment: Linear SVM &nbsp;|&nbsp; Category: LightGBM
                    </p>
                </div>
                """, unsafe_allow_html=True)

            except FileNotFoundError:
                st.error("Model files not found. Make sure `best_sentiment_model.pkl` and `best_category_model.pkl` are in the same folder as `app.py`.")

    st.markdown("---")
    st.markdown('<p class="section-label">Batch Classification</p>', unsafe_allow_html=True)
    st.markdown("### Classify Multiple Reviews at Once")
    uploaded = st.file_uploader("Upload a CSV file with a column named `Review Text`", type=['csv'])
    if uploaded:
        try:
            sent_bundle, cat_bundle = load_models()
            sw, lem = get_nlp_tools()
            df_up = pd.read_csv(uploaded)
            if 'Review Text' not in df_up.columns:
                st.error("CSV must have a column named exactly `Review Text`.")
            else:
                with st.spinner(f"Classifying {len(df_up)} reviews…"):
                    results = [predict(str(r), sent_bundle, cat_bundle, sw, lem)
                               for r in df_up['Review Text'].fillna('')]
                    df_up['Predicted Sentiment'] = [r[0] for r in results]
                    df_up['Predicted Category']  = [r[1] for r in results]
                st.success(f"Done — {len(df_up)} reviews classified.")
                st.dataframe(df_up[['Review Text', 'Predicted Sentiment', 'Predicted Category']].head(50))
                csv_out = df_up.to_csv(index=False).encode()
                st.download_button("⬇ Download Results", csv_out, "classified_reviews.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Dashboard":
    st.markdown('<p class="section-label">Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown("### Women's E-Commerce Clothing Reviews")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi-card"><p class="kpi-value">22,641</p><p class="kpi-label">Total Reviews</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-card"><p class="kpi-value">4.18★</p><p class="kpi-label">Average Rating</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-card"><p class="kpi-value">77.1%</p><p class="kpi-label">Positive Sentiment</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-card"><p class="kpi-value">81.9%</p><p class="kpi-label">Would Recommend</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Reviews by Complaint Category**")
        cats  = list(STATS['category_counts'].keys())
        vals  = list(STATS['category_counts'].values())
        colors = ['#7c3aed','#9333ea','#a855f7','#c084fc','#ddd6fe','#ede9fe']
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(cats, vals, color=colors)
        ax.set_xlabel('Number of Reviews', fontsize=10)
        for bar, val in zip(ax.patches, vals):
            ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**Sentiment Distribution**")
        sent_labels = list(STATS['sentiment_counts'].keys())
        sent_vals   = list(STATS['sentiment_counts'].values())
        sent_colors = ['#10b981', '#f59e0b', '#ef4444']
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.pie(sent_vals, labels=sent_labels, colors=sent_colors,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        fig2.patch.set_facecolor('white')
        st.pyplot(fig2)
        plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Rating Distribution**")
        r_labels = [f'★{k}' for k in STATS['rating_counts'].keys()]
        r_vals   = list(STATS['rating_counts'].values())
        r_colors = ['#ef4444','#f97316','#f59e0b','#84cc16','#10b981']
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.bar(r_labels, r_vals, color=r_colors, edgecolor='white')
        ax3.set_ylabel('Reviews', fontsize=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        for bar, val in zip(ax3.patches, r_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                     f'{val:,}', ha='center', fontsize=9)
        fig3.patch.set_facecolor('white')
        st.pyplot(fig3)
        plt.close()

    with c4:
        st.markdown("**Avg Rating by Department**")
        depts = list(STATS['dept_avg_rating'].keys())
        d_vals = list(STATS['dept_avg_rating'].values())
        d_colors = ['#ef4444' if v < 4.0 else '#f59e0b' if v < 4.2 else '#10b981'
                    for v in d_vals]
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.barh(depts, d_vals, color=d_colors)
        ax4.set_xlim(3.5, 4.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        for bar, val in zip(ax4.patches, d_vals):
            ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}★', va='center', fontsize=9)
        fig4.patch.set_facecolor('white')
        st.pyplot(fig4)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.markdown('<p class="section-label">Comparative Analysis</p>', unsafe_allow_html=True)
    st.markdown("### All Models — Side by Side")
    st.markdown("Six classical ML models plus DistilBERT (transformer) were trained and evaluated on two tasks.")

    tab1, tab2 = st.tabs(["Sentiment Classification", "Category Classification"])

    with tab1:
        sent_data = STATS['model_results']['sentiment']
        df_sent = pd.DataFrame(sent_data).T.reset_index()
        df_sent.columns = ['Model', 'Accuracy', 'F1 (weighted)']
        df_sent = df_sent.sort_values('F1 (weighted)', ascending=False).reset_index(drop=True)
        df_sent.index = df_sent.index + 1
        df_sent['Accuracy'] = df_sent['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_sent['F1 (weighted)'] = df_sent['F1 (weighted)'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_sent, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        models_s = list(sent_data.keys())
        f1s_s = [sent_data[m]['f1'] for m in models_s]
        bar_c = ['#f59e0b' if v == max(f1s_s) else '#3b82f6' if m == 'DistilBERT' else '#7c3aed'
                 for m, v in zip(models_s, f1s_s)]
        ax.bar(models_s, f1s_s, color=bar_c, edgecolor='white')
        ax.set_ylim(0.55, 1.0)
        ax.set_ylabel('F1 Score (weighted)')
        ax.set_title('Sentiment — F1 by Model', fontweight='bold')
        ax.set_xticklabels(models_s, rotation=15, ha='right')
        for i, val in enumerate(f1s_s):
            ax.text(i, val + 0.008, f'{val:.3f}', ha='center', fontsize=9)
        patches = [
            mpatches.Patch(color='#f59e0b', label='Best model'),
            mpatches.Patch(color='#3b82f6', label='DistilBERT'),
            mpatches.Patch(color='#7c3aed', label='Classical model'),
        ]
        ax.legend(handles=patches, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close()

    with tab2:
        cat_data = STATS['model_results']['category']
        df_cat = pd.DataFrame(cat_data).T.reset_index()
        df_cat.columns = ['Model', 'Accuracy', 'F1 (weighted)']
        df_cat = df_cat.sort_values('F1 (weighted)', ascending=False).reset_index(drop=True)
        df_cat.index = df_cat.index + 1
        df_cat['Accuracy'] = df_cat['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_cat['F1 (weighted)'] = df_cat['F1 (weighted)'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_cat, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        models_c = list(cat_data.keys())
        f1s_c = [cat_data[m]['f1'] for m in models_c]
        bar_c2 = ['#f59e0b' if v == max(f1s_c) else '#7c3aed' for v in f1s_c]
        ax2.bar(models_c, f1s_c, color=bar_c2, edgecolor='white')
        ax2.set_ylim(0.55, 1.05)
        ax2.set_ylabel('F1 Score (weighted)')
        ax2.set_title('Category — F1 by Model', fontweight='bold')
        ax2.set_xticklabels(models_c, rotation=15, ha='right')
        for i, val in enumerate(f1s_c):
            ax2.text(i, val + 0.005, f'{val:.3f}', ha='center', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        fig2.patch.set_facecolor('white')
        st.pyplot(fig2)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Insights":
    st.markdown('<p class="section-label">Operational Intelligence</p>', unsafe_allow_html=True)
    st.markdown("### Actionable Insights for Management")
    st.markdown("What the data is telling the business — translated into decisions.")

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">// TOP COMPLAINT THEME</div>
        <b>Sizing & Fit accounts for 79.9% of all tagged reviews.</b> This is the single highest-priority area.
        Invest in detailed size guides, measurement charts, and user-generated fit feedback (e.g. reviewer height and size bought).
    </div>
    <div class="insight-card amber">
        <div class="insight-title">// AT-RISK DEPARTMENT</div>
        <b>Trend department has the lowest average rating at 3.84★</b> — well below the 4.18★ overall average.
        This department needs a product quality audit. It may be a sourcing problem or a mismatch between product descriptions and reality.
    </div>
    <div class="insight-card green">
        <div class="insight-title">// STRONG PERFORMER</div>
        <b>Bottoms and Intimates lead with 4.27–4.28★.</b> Study what works there — fit consistency, fabric quality —
        and apply those lessons to underperforming departments.
    </div>
    <div class="insight-card red">
        <div class="insight-title">// NEGATIVE REVIEW ALERT</div>
        <b>821 reviews are 1-star.</b> An automated classifier deployed on incoming reviews can flag these in real time
        for customer service follow-up, rather than relying on manual checking.
    </div>
    <div class="insight-card">
        <div class="insight-title">// BEST MODEL FOR DEPLOYMENT</div>
        <b>LightGBM for category (F1 = 0.970) and DistilBERT for sentiment (F1 = 0.839).</b>
        Both are lightweight at inference time and can be wrapped in an API with no GPU required for serving.
    </div>
    <div class="insight-card amber">
        <div class="insight-title">// RECOMMENDATION RATE</div>
        <b>81.9% of customers say they would recommend the product.</b> A referral or loyalty programme
        could be built directly on this signal. Target the 18.1% non-recommenders with follow-up surveys.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sentiment breakdown**")
        for label, count in STATS['sentiment_counts'].items():
            pct = count / STATS['total_reviews'] * 100
            st.progress(int(pct), text=f"{label}: {count:,} ({pct:.1f}%)")
    with col2:
        st.markdown("**Category breakdown**")
        total_cat = sum(STATS['category_counts'].values())
        for label, count in STATS['category_counts'].items():
            pct = count / total_cat * 100
            st.progress(int(pct), text=f"{label}: {count:,} ({pct:.1f}%)")
