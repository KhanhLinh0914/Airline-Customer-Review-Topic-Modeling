# Airline-Customer-Review-Topic-Modeling

# Airline Customer Review Topic Modeling
 

## Overview
This project applies **Natural Language Processing (NLP) and topic modeling** to 23,171 airline customer reviews (sourced from Skytrax) to uncover the key themes driving passenger satisfaction and dissatisfaction. Rather than manually reading thousands of reviews, the project uses unsupervised topic modeling to surface recurring patterns at scale, turning unstructured text feedback into actionable business insight.

**Team:** Neha Kataria, Thi Khanh Linh Pham (Sylvia), Muskan, Ke Ping Lo, Karyn Denise Pang, Gull Qazi

## Business Problem
Airlines receive large volumes of written customer reviews containing valuable feedback, but manually reviewing them is time-consuming and makes it hard to identify recurring issues. This project uses topic modeling to automatically surface the main themes in reviews: delays, refunds, baggage handling, seating, cabin service, so airlines can prioritize fixes and reinforce what's working.

## Dataset
- 23,171 airline reviews, 20 variables (airline name, overall rating, traveller type, comfort/staff/ground-service ratings, value for money, recommendation status, review text, etc.)
- Several columns had substantial missing data (e.g., in-flight entertainment, Wi-Fi/connectivity)
- After cleaning and de-duplication: **23,046 unique reviews** used for text analysis

## Tech Stack
- **Language:** Python  
- **NLP:** NLTK (tokenization, stopwords, lemmatization), regex/string cleaning
- **Feature extraction:** scikit-learn `TfidfVectorizer` (max 5,000 features)
- **Topic modeling:** scikit-learn `LatentDirichletAllocation` (LDA), scikit-learn `NMF`
- **Model evaluation:** Gensim (`CoherenceModel`, `Dictionary`) for C_v coherence and topic diversity
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Environment:** Jupyter Notebook

## Methodology
1. **Data Cleaning**: dropped high-missingness columns (review title, verified Wi-Fi/entertainment/food), renamed columns, imputed numeric ratings with median, imputed missing categorical values, removed duplicate reviews
2. **Text Preprocessing (NLP)**: lowercased, removed punctuation, tokenized, removed stopwords, lemmatized to base word forms
3. **Feature Engineering**: TF-IDF vectorization (max 5,000 features, rare/overly common terms excluded)
4. **Topic Modeling**: fit both LDA and NMF with k=5 topics on the same TF-IDF matrix
5. **Model Evaluation**: compared LDA vs. NMF using C_v coherence score, topic diversity, and a qualitative stress test (5 manually written reviews with known themes)

## Results

### Model Comparison
| Model | Coherence (C_v) | Topic Diversity | Notes |
|---|---|---|---|
| LDA | Lower | Lower | Topic redundancy; Topic 4 (business class/meal service) under-represented at 0.7% of reviews; misclassified a baggage review as a delay issue in the stress test |
| **NMF (selected)** | **0.5090** | **0.8133** | Balanced topics (all >14% share); correctly separated baggage vs. delay themes in the stress test (0.775 top-topic confidence) |

### NMF Topics (Final Model)
| Topic | Share of Reviews | Avg. Rating |
|---|---|---|
| Positive Cabin Experience | 29.0% | 5.55 (highest) |
| Flight Delays, Cancellations & Schedule Disruption | 21.6% | 1.63 |
| Fees & Customer Support/Refund Escalations | 18.1% | 1.52 (lowest) |
| Seat Comfort & Cabin Class Features | 16.7% | 3.13 |
| Airport Boarding & Baggage Handling | 14.7% | 2.72 |

## Business Insights
- **Positive Cabin Experience** is the strongest area of passenger satisfaction: cabin service, cleanliness, and staff interactions drive the highest ratings.
- **Flight disruptions and customer support/refunds** are the biggest pain points, with the lowest average ratings.
- **Boarding/baggage handling** and **seat comfort** are moderate-rated areas with clear room for improvement.
- Recommendation: airlines should prioritize operational reliability (delays, disruption communication) and customer support/refund processes, while maintaining strengths in cabin experience. 

## Limitations
- Analysis is based only on customers who chose to leave a review (self-selection bias)
- Fixed topic count (k=5) may not capture every distinct issue; reviews can span multiple themes at once
- Topic labeling/interpretation is inherently subjective
- Preprocessing choices (e.g., word removal) can lose useful context

## Future Work
- Test a range of topic counts (k) rather than fixing k=5
- Incorporate n-grams (e.g., "flight delay", "customer service") to capture multi-word phrases
- Combine topic modeling with sentiment analysis
- Compare topic prevalence across airlines, traveller types, seat classes
- Explore temporal trends in topic prevalence to catch emerging issues
- Evaluate more advanced models (e.g., BERTopic, transformer-based topic models)

## Repository Structure
```
├── BI_FinalProject_Group_3.ipynb                 # Main notebook: cleaning, NLP preprocessing, TF-IDF, LDA/NMF modeling, evaluation
├── Business_Intelligence_-_Final_Project.docx    # Full final report
├── Airline_reviews.csv                           # Raw airline customer review dataset (Skytrax, 23,171 reviews)
└── README.md
```

## Dataset Source
Airline customer review dataset sourced from Skytrax (23,171 reviews).
