import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Buyer Segmentation & Investment Profiling", layout="wide")

# ------------------------------------------------------------------
# DATA LOADING + PIPELINE (cached so it only runs once, not on every filter change)
# ------------------------------------------------------------------

@st.cache_data
def load_and_process(clients_path, properties_path):
    c = pd.read_csv(clients_path)
    p = pd.read_csv(properties_path)

    # --- DOB parsing: dash rows = DD-MM-YYYY, slash rows = MM-DD-YYYY (forced by data) ---
    def parse_dob(s):
        if '/' in s:
            return datetime.strptime(s, '%m/%d/%Y')
        else:
            return datetime.strptime(s, '%d-%m-%Y')
    c['dob_parsed'] = c['date_of_birth'].apply(parse_dob)
    today_ref = datetime(2025, 12, 31)
    c['age'] = c['dob_parsed'].apply(
        lambda d: today_ref.year - d.year - ((today_ref.month, today_ref.day) < (d.month, d.day))
    )

    # --- Encoding: binary for true 2-value fields, one-hot for 3+ value fields ---
    c['client_type_enc'] = c['client_type'].map({'Individual': 0, 'Company': 1})
    c['gender_enc'] = c['gender'].map({'F': 0, 'M': 1})
    c['loan_applied_enc'] = c['loan_applied'].map({'No': 0, 'Yes': 1})
    c['acquisition_purpose_enc'] = c['acquisition_purpose'].map({'Home': 0, 'Investment': 1})

    c_encoded = pd.get_dummies(c, columns=['country', 'referral_channel'], prefix=['country', 'referral'])

    feature_cols = (
        ['client_type_enc', 'gender_enc', 'loan_applied_enc', 'acquisition_purpose_enc', 'age', 'satisfaction_score']
        + [col for col in c_encoded.columns if col.startswith('country_') or col.startswith('referral_')]
    )
    X = c_encoded[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['age', 'satisfaction_score']] = scaler.fit_transform(X[['age', 'satisfaction_score']])

    # --- Clustering: k=4 (chosen for interpretability - silhouette scores 0.14-0.19
    # across all tested k, below the 0.25 threshold for strong cluster separation) ---
    kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    c['segment'] = kmeans_final.fit_predict(X_scaled)

    # Descriptive labels based on what actually differentiates segments (age/satisfaction) -
    # NOT business personas like "Global Investors", since categorical fields showed no
    # meaningful separation and investment behavior doesn't differ by segment
    seg_profile = c.groupby('segment')[['age', 'satisfaction_score']].mean()
    age_median = seg_profile['age'].median()
    sat_median = seg_profile['satisfaction_score'].median()

    def label_segment(row):
        age_lbl = "Older" if row['age'] >= age_median else "Younger"
        sat_lbl = "High Satisfaction" if row['satisfaction_score'] >= sat_median else "Low Satisfaction"
        return f"{age_lbl}, {sat_lbl}"

    seg_labels = seg_profile.apply(label_segment, axis=1).to_dict()
    c['segment_label'] = c['segment'].map(seg_labels)

    # --- Phase 2: properties join, per-client aggregation, Available listings excluded ---
    p_sold = p[p['listing_status'] == 'Sold'].copy()
    p_sold['sale_price_clean'] = (
        p_sold['sale_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    )
    p_sold['transaction_date_parsed'] = pd.to_datetime(p_sold['transaction_date'], format='%m-%d-%Y')

    client_agg = p_sold.groupby('client_ref').agg(
        num_purchases=('listing_id', 'count'),
        total_spend=('sale_price_clean', 'sum'),
        avg_floor_area=('floor_area_sqft', 'mean'),
    ).reset_index()

    dominant_type = (
        p_sold.groupby('client_ref')['unit_category']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={'unit_category': 'dominant_property_type'})
    )
    client_agg = client_agg.merge(dominant_type, on='client_ref')

    overlay = c.merge(client_agg, left_on='client_id', right_on='client_ref', how='left')

    return overlay


# ISO-3 codes for the world map (Module 3) - covers all 10 countries in this dataset
ISO3_MAP = {
    'USA': 'USA', 'UK': 'GBR', 'Canada': 'CAN', 'Germany': 'DEU', 'France': 'FRA',
    'Belgium': 'BEL', 'Mexico': 'MEX', 'Russia': 'RUS', 'Australia': 'AUS', 'Denmark': 'DNK'
}

# ------------------------------------------------------------------
# LOAD DATA - change these paths to wherever your CSVs are
# ------------------------------------------------------------------
CLIENTS_PATH = 'clients.csv'
PROPERTIES_PATH = 'properties.csv'

df = load_and_process(CLIENTS_PATH, PROPERTIES_PATH)

# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.header("Filters")

countries = st.sidebar.multiselect("Country", sorted(df['country'].unique()), default=sorted(df['country'].unique()))
regions = st.sidebar.multiselect("Region", sorted(df['region'].unique()), default=sorted(df['region'].unique()))
purposes = st.sidebar.multiselect("Acquisition Purpose", sorted(df['acquisition_purpose'].unique()), default=sorted(df['acquisition_purpose'].unique()))
client_types = st.sidebar.multiselect("Client Type", sorted(df['client_type'].unique()), default=sorted(df['client_type'].unique()))

filtered = df.copy()
if countries:
    filtered = filtered[filtered['country'].isin(countries)]
if regions:
    filtered = filtered[filtered['region'].isin(regions)]
if purposes:
    filtered = filtered[filtered['acquisition_purpose'].isin(purposes)]
if client_types:
    filtered = filtered[filtered['client_type'].isin(client_types)]

st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)} clients**")

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("Buyer Segmentation & Investment Profiling")
st.caption(
    "Segments are based on age and satisfaction score, the two features that showed "
    "meaningful statistical separation (ANOVA p<0.001). Other client attributes "
    "(client_type, gender, loan status, country, referral channel) did not differentiate "
    "segments in a practically significant way. Investment behavior (spend, purchase "
    "frequency, property type) does not differ meaningfully across segments (all "
    "correlations with age/satisfaction below 0.1)."
)

# ------------------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ------------------------------------------------------------------
st.header("1. Buyer Segmentation Overview")

col1, col2 = st.columns([1, 1])
with col1:
    seg_counts = filtered['segment_label'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig = px.pie(seg_counts, names='Segment', values='Count', title='Segment Distribution')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.bar(seg_counts, x='Segment', y='Count', title='Segment Sizes', color='Segment')
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ------------------------------------------------------------------
st.header("2. Investor Behavior Dashboard")

inv_summary = filtered.groupby('segment_label').agg(
    avg_num_purchases=('num_purchases', 'mean'),
    avg_total_spend=('total_spend', 'mean'),
    avg_floor_area=('avg_floor_area', 'mean'),
).reset_index()

col3, col4, col5 = st.columns(3)
with col3:
    fig3 = px.bar(inv_summary, x='segment_label', y='avg_num_purchases', title='Avg Purchases by Segment')
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    fig4 = px.bar(inv_summary, x='segment_label', y='avg_total_spend', title='Avg Total Spend by Segment')
    st.plotly_chart(fig4, use_container_width=True)
with col5:
    fig5 = px.bar(inv_summary, x='segment_label', y='avg_floor_area', title='Avg Floor Area by Segment')
    st.plotly_chart(fig5, use_container_width=True)

st.caption(
    "Note: differences shown above are small relative to overall variation and were "
    "either not statistically significant or had negligible effect sizes in formal testing."
)

# ------------------------------------------------------------------
# MODULE 3: GEOGRAPHIC BUYER ANALYSIS (world map)
# ------------------------------------------------------------------
st.header("3. Geographic Buyer Analysis")

country_counts = filtered['country'].value_counts().reset_index()
country_counts.columns = ['country', 'clients']
country_counts['iso3'] = country_counts['country'].map(ISO3_MAP)

fig_map = px.choropleth(
    country_counts,
    locations='iso3',
    color='clients',
    hover_name='country',
    color_continuous_scale='Blues',
    title='Client Distribution by Country'
)
fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True))
st.plotly_chart(fig_map, use_container_width=True)

col6, col7 = st.columns([1, 1])
with col6:
    fig6 = px.bar(country_counts, x='country', y='clients', title='Clients by Country (bar view)')
    st.plotly_chart(fig6, use_container_width=True)

with col7:
    region_counts = filtered['region'].value_counts().reset_index().head(15)
    region_counts.columns = ['Region', 'Clients']
    fig7 = px.bar(region_counts, x='Region', y='Clients', title='Top 15 Regions by Client Count')
    st.plotly_chart(fig7, use_container_width=True)

# ------------------------------------------------------------------
# MODULE 4: SEGMENT INSIGHTS PANEL (tables + mini charts)
# ------------------------------------------------------------------
st.header("4. Segment Insights Panel")

# Summary table across all segments at a glance
summary_table = filtered.groupby('segment_label').agg(
    n_clients=('client_id', 'count'),
    avg_age=('age', 'mean'),
    avg_satisfaction=('satisfaction_score', 'mean'),
    avg_purchases=('num_purchases', 'mean'),
    avg_total_spend=('total_spend', 'mean'),
).round(2).reset_index().rename(columns={'segment_label': 'Segment'})

st.subheader("Segment Summary Table")
st.dataframe(summary_table, use_container_width=True, hide_index=True)

st.subheader("Segment Detail")
for seg in sorted(filtered['segment_label'].unique()):
    sub = filtered[filtered['segment_label'] == seg]
    with st.expander(f"{seg}  (n={len(sub)})"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Age", f"{sub['age'].mean():.1f}")
        c2.metric("Avg Satisfaction", f"{sub['satisfaction_score'].mean():.2f}")
        c3.metric("Avg Purchases", f"{sub['num_purchases'].mean():.2f}")
        c4.metric("Avg Total Spend", f"${sub['total_spend'].mean():,.0f}")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            ct = sub['client_type'].value_counts().reset_index()
            ct.columns = ['Client Type', 'Count']
            fig_ct = px.pie(ct, names='Client Type', values='Count', title='Client Type', hole=0.4)
            fig_ct.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_ct, use_container_width=True)
        with mc2:
            ap = sub['acquisition_purpose'].value_counts().reset_index()
            ap.columns = ['Purpose', 'Count']
            fig_ap = px.pie(ap, names='Purpose', values='Count', title='Acquisition Purpose', hole=0.4)
            fig_ap.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_ap, use_container_width=True)
        with mc3:
            top_countries = sub['country'].value_counts().reset_index().head(5)
            top_countries.columns = ['Country', 'Count']
            fig_tc = px.bar(top_countries, x='Country', y='Count', title='Top Countries')
            fig_tc.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_tc, use_container_width=True)
