# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

Client segmentation and investment behavior analysis for a real estate platform, built as part of a Unified Mentor data analyst internship project (in collaboration with Parcl Co. Limited).

**Live Dashboard:** [deepchovatiya-real-estate-analytics.streamlit.app](https://deepchovatiya-real-estate-analytics.streamlit.app)

## Project Overview

This project applies K-Means and Hierarchical clustering to segment 2,000 real estate clients based on demographic and behavioral attributes, then tests whether those segments correspond to distinct investment behavior using a linked dataset of 10,000 property listings.

**Key finding:** the four resulting segments are driven almost entirely by client age and satisfaction score. Silhouette scores across all tested cluster counts (k = 2–8) ranged from 0.14–0.19, below the 0.25 threshold generally associated with meaningful cluster separation — a limitation that is stated directly in the paper rather than glossed over. Investment behavior (purchase frequency, total spend, property size) does not differ meaningfully across these segments (all correlations below 0.1). The most reliable, actionable finding in the dataset turned out to be a ~40% decline in monthly sales volume beginning mid-2024, identified during EDA.

## Repository Contents

| File | Description |
|---|---|
| `clients.csv` | Raw client dataset (2,000 records) |
| `properties.csv` | Raw property listing dataset (10,000 records) |
| `buyer_segmentation_analysis.ipynb` | Full analysis notebook: data cleaning, EDA, feature engineering, K-Means/Hierarchical/K-Prototypes clustering, statistical validation, investment overlay |
| `dashboard_app.py` | Streamlit dashboard (segmentation overview, investor behavior, geographic map, segment insights) with live filters |
| `research_paper.docx` | Full write-up: methodology, EDA, clustering results, investment profiling, limitations, and recommendations |
| `requirements.txt` | Python dependencies |

## Methodology Summary

- **Data cleaning:** resolved mixed date formats in `date_of_birth` (dual DD-MM-YYYY / MM-DD-YYYY parsing, disambiguated using value constraints in the data) and `transaction_date`; converted currency strings to numeric.
- **Feature engineering:** binary encoding for true two-value fields, one-hot encoding for higher-cardinality nominal fields, `region` excluded from modeling due to high cardinality (57 values) but retained for dashboard filtering.
- **Clustering:** K-Means (k=4, selected for interpretability — not because it scored best on silhouette) cross-validated against Hierarchical clustering (Adjusted Rand Index = 0.357) and K-Prototypes with Gower distance (used to confirm weak separation wasn't a one-hot-encoding artifact).
- **Statistical validation:** ANOVA and Chi-square testing confirm which features actually drive segment assignment (age, satisfaction) versus which contribute negligibly (gender, loan status, country, referral channel).
- **Investment overlay:** properties joined per-client (Sold listings only), aggregated to purchase count / total spend / average floor area, tested against segments via ANOVA and correlation.

## Dashboard

Run locally:
```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
```

## Notes on Data

This is a synthetic dataset provided for the internship project. A few properties of the data are called out explicitly in the paper's Limitations section — including a near-uniform satisfaction score distribution (atypical of real survey data) and 100% buyer conversion across all clients — since they affect how the findings should be interpreted.
