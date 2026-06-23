# CineMood AI Presentation Outline

This document outlines the PowerPoint presentation structure for **CineMood AI: An Emotion-Aware Hybrid Movie Recommendation System**.

---

## Slide 1: Title Slide
* **Title**: CineMood AI
* **Subtitle**: An Emotion-Aware Hybrid Movie Recommendation System Using Sentiment Analysis, NLP, Transformer Models, and Real-Time Movie Data Integration
* **Presenter**: Antigravity AI
* **Date**: June 2026

---

## Slide 2: The Core Problem
* **Observation**: Current recommenders (Netflix, Prime Video, etc.) are "mood-blind." They ignore user emotional states, focusing strictly on historical ratings and static genres.
* **Impact**: Suggesting dark crime thrillers to a user seeking comfort after a stressful day, or vice versa, degrading user satisfaction.
* **Solution**: An emotion-aware engine that translates natural language descriptions of feelings into tailored movie moods.

---

## Slide 3: CineMood AI Architecture
* **Diagram/Workflow**:
  1. **User Text Input** -> Analyzed for Sentiment and Emotions.
  2. **NLP Engine** -> Produces a normalized 8-Mood probability vector.
  3. **Hybrid Recommendation Engine** -> Computes joint scores:
     - 40% Mood Match Score
     - 30% Content Similarity (TF-IDF Soup)
     - 30% Collaborative Score (SVD Matrix Factorization)
  4. **TMDB Real-Time Layer** -> Fetches posters, details, and trailer links dynamically.
  5. **Streamlit UI** -> Displays a beautiful responsive grid of recommendation cards.

---

## Slide 4: Unified Movie Knowledge Base
* **Data Sources**:
  * **MovieLens 32M**: 32M Ratings used for collaborative latent signals and popularity.
  * **TMDB metadata**: Cast, director, overviews, keywords, and poster paths.
* **Integration Strategy**:
  * Merged using movie links tables.
  * Pre-aggregated rating metadata (mean ratings & count) to prevent runtime lags.
  * Extracted structural features (JSON keywords, directors, actor lists).
  * Generated 21-feature master movie CSV of 87,585 titles in under 24 seconds.

---

## Slide 5: Sentiment & Emotion Classifiers
* **Sentiment Analysis Model**:
  * Dataset: 50,000 IMDb movie reviews.
  * Preprocessing: HTML strip, tokenization, stopword removal.
  * Model: **Logistic Regression** over TF-IDF bigrams.
  * Accuracy: **89.76%** (validated against Multinomial Naive Bayes at 86.82%).
* **Emotion & Mood Mapping Model**:
  * Extracts 6 primary emotions (Joy, Sadness, Anger, Fear, Surprise, Love).
  * Implements linear mapping matrix to project basic emotions onto 8 target moods (Happy, Sad, Excited, Relaxed, Inspirational, Romantic, Thoughtful, Suspenseful).
  * Embedded lexical backup mapping for complete configuration-free robustness.

---

## Slide 6: Collaborative & Content Recommendations
* **Collaborative Filtering SVD**:
  * Built via **scikit-surprise** library.
  * Downsampled 32M ratings (filtering $\ge 150$ ratings/user and ratings/movie) to 800,000 ratings.
  * Matrix factorization training time: **21.47 seconds**.
  * Validation Metrics: **RMSE = 0.8801**, **MAE = 0.6738**.
* **Content Similarity**:
  * Computes cosine similarity of TF-IDF "soup" descriptors.
  * Executes dynamically over filtered candidate lists to eliminate RAM bottlenecks.

---

## Slide 7: The Hybrid Blending Engine
* **Formula**:
  $$\text{Score}(u, m) = 0.40 \cdot \text{Mood}(m) + 0.30 \cdot \text{Content}(m) + 0.30 \cdot \text{SVD}(u, m)$$
* **Cold-Start Handling**:
  * If SVD has no user history, it automatically shifts collaborative weights to normalized popularity and movie average ratings.
  * If no target movie is searched, content weights are shifted to Mood Match (60% Mood / 40% Collaborative).

---

## Slide 8: Real-Time TMDB Layer
* **Dynamic Media Loader**:
  * Fetches real-time movie poster paths from `https://image.tmdb.org/t/p/w500`.
  * Grabs YouTube video trailers from TMDB API movie queries.
* **Fail-Safe Fallbacks**:
  * Works out-of-the-box without API keys by using local poster paths and triggering automated YouTube trailer searches.
  * Implements LRU caching to eliminate query latency.

---

## Slide 9: Premium Streamlit User Experience
* **Styling & UX Highlights**:
  * Custom stylesheet (`app/style.css`) utilizing Outfit typography and deep dark gradients.
  * Glassmorphic search cards and hover-responsive movie cards with glowing borders.
  * Interactive Analytics dashboard displaying Plotly charts of genre frequency, session history mood pies, and top MovieLens movies.
  * Expandable modal overlays showing cast, crew, overview details, and embedded trailer feeds.

---

## Slide 10: Conclusion & Next Steps
* **Successes**:
  * Successfully merged massive datasets and trained complex models locally in under 1 minute.
  * Recommends contextual mood-matched movies with sub-second execution speeds.
  * Fully verified using unit tests.
* **Next Steps**:
  * Fine-tuning Deep Transformer models (e.g. RoBERTa) on movie-specific tag alignments.
  * Graph Neural Networks (GNNs) for heterogeneous movie representation learning.
