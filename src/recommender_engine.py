import os
import pandas as pd
import numpy as np
import joblib
import time
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

import src.database as db

class CineMoodRecommender:
    def __init__(self, master_df_path="data/master_movies.csv", model_dir="models", use_semantic=False):
        self.master_df_path = master_df_path
        self.model_dir = model_dir
        self.df = None
        self.svd_model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.genre_mood_weights = None
        
        # Disabled transformers permanently to run light on the CPU
        self.use_semantic = False
        
        # Load dataset
        self.load_data()
        # Initialize mood mappings
        self.init_mood_mappings()
        # Load SVD model if available
        self.load_svd_model()
        # Precompute movie moods
        self.compute_movie_moods()
        # Precompute soup and TF-IDF for fast content search
        self.init_content_features()
        # Initialize database
        db.init_db()
        
    def load_data(self):
        print(f"Loading master movies dataset from {self.master_df_path}...")
        if not os.path.exists(self.master_df_path):
            raise FileNotFoundError(f"Master movies dataset not found. Run src/data_pipeline.py first.")
        self.df = pd.read_csv(self.master_df_path)
        self.df["movieId"] = self.df["movieId"].astype(int)
        self.df["tmdbId"] = self.df["tmdbId"].fillna(0).astype(int)
        self.df["popularity"] = self.df["popularity"].fillna(0.0).astype(float)
        self.df["rating_count"] = self.df["rating_count"].fillna(0).astype(int)
        self.df["avg_rating"] = self.df["avg_rating"].fillna(2.5).astype(float)
        print(f"Loaded {len(self.df):,} movies.")
        
    def init_mood_mappings(self):
        # Genre weights for the 8 target moods
        self.genre_mood_weights = {
            "Action": {"Suspenseful": 0.4, "Excited": 0.4, "Happy": 0.2},
            "Adventure": {"Excited": 0.5, "Inspirational": 0.3, "Happy": 0.2},
            "Animation": {"Happy": 0.6, "Relaxed": 0.2, "Excited": 0.2},
            "Children": {"Happy": 0.7, "Relaxed": 0.3},
            "Comedy": {"Happy": 0.8, "Relaxed": 0.2},
            "Crime": {"Suspenseful": 0.6, "Thoughtful": 0.4},
            "Documentary": {"Thoughtful": 0.8, "Inspirational": 0.2},
            "Drama": {"Thoughtful": 0.5, "Sad": 0.3, "Romantic": 0.2},
            "Fantasy": {"Excited": 0.3, "Happy": 0.3, "Thoughtful": 0.4},
            "Film-Noir": {"Suspenseful": 0.5, "Thoughtful": 0.5},
            "Horror": {"Suspenseful": 0.8, "Thoughtful": 0.2},
            "IMAX": {"Excited": 0.8, "Inspirational": 0.2},
            "Musical": {"Happy": 0.6, "Romantic": 0.3, "Relaxed": 0.1},
            "Mystery": {"Suspenseful": 0.7, "Thoughtful": 0.3},
            "Romance": {"Romantic": 0.8, "Relaxed": 0.1, "Happy": 0.1},
            "Sci-Fi": {"Thoughtful": 0.6, "Excited": 0.4},
            "Thriller": {"Suspenseful": 0.7, "Excited": 0.3},
            "War": {"Thoughtful": 0.6, "Sad": 0.4},
            "Western": {"Suspenseful": 0.4, "Excited": 0.4, "Thoughtful": 0.2}
        }
        
    def load_svd_model(self):
        svd_path = os.path.join(self.model_dir, "svd_model.pkl")
        if os.path.exists(svd_path):
            try:
                self.svd_model = joblib.load(svd_path)
                print("Collaborative SVD model loaded successfully!")
            except Exception as e:
                print(f"Error loading SVD model: {e}. Collaborative score will fall back to popularity.")
        else:
            print("SVD model not found. Collaborative will fallback to popularity/average ratings.")
            
    def compute_movie_moods(self):
        print("Computing movie mood profiles based on genres and keywords...")
        moods = ["Happy", "Sad", "Excited", "Relaxed", "Inspirational", "Romantic", "Thoughtful", "Suspenseful"]
        
        # Initialize mood score columns
        for m in moods:
            self.df[f"mood_{m}"] = 0.0
            
        def get_movie_mood_vector(row):
            vector = {m: 0.0 for m in moods}
            genres_list = str(row["genres"]).split("|")
            for genre in genres_list:
                if genre in self.genre_mood_weights:
                    for m, w in self.genre_mood_weights[genre].items():
                        vector[m] += w
                        
            keywords_and_tags = str(row["extracted_keywords"]).lower() + " " + str(row["tags_list"]).lower()
            keyword_boosts = {
                "Sad": ["sad", "depress", "heartbreak", "tragedy", "loss", "mourn", "grief", "melancholy", "tear", "lonely", "unhappy", "gloomy"],
                "Romantic": ["love", "romance", "marriage", "couple", "kiss", "date", "affection", "sweetheart", "romantic", "dating", "beloved"],
                "Suspenseful": ["kill", "murder", "detective", "mystery", "scary", "ghost", "thrill", "suspense", "terror", "creepy", "dark", "crime", "investigat", "spy", "agent"],
                "Happy": ["funny", "comedy", "laugh", "hilarious", "family", "kids", "celebrate", "joy", "cheerful", "humor", "silly", "warm", "fun"],
                "Inspirational": ["inspire", "triumph", "courage", "overcome", "hope", "dream", "motivation", "determination", "success", "uplift"],
                "Excited": ["action", "explosion", "fight", "superhero", "chase", "adventure", "battle", "epic", "combat", "warrior", "heroic", "monster"],
                "Relaxed": ["calm", "cozy", "peaceful", "gentle", "slow", "quiet", "serene", "chill", "comfort", "sooth"],
                "Thoughtful": ["philosoph", "existential", "psycholog", "mind", "science", "documentary", "intellectual", "deep", "complex", "brainy", "puzzl", "history", "biograph"]
            }
            
            for m, terms in keyword_boosts.items():
                for term in terms:
                    if term in keywords_and_tags:
                        vector[m] += 0.25
                        
            total = sum(vector.values())
            if total > 0:
                vector = {m: score / total for m, score in vector.items()}
            else:
                vector = {m: 1.0 / len(moods) for m in moods}
            return pd.Series(vector)
            
        mood_vectors = self.df.apply(get_movie_mood_vector, axis=1)
        for m in moods:
            self.df[f"mood_{m}"] = mood_vectors[m]
        print("Movie mood profiles computed successfully.")
        
    def init_content_features(self):
        print("Initializing content metadata features (soup)...")
        def create_soup(row):
            genres_clean = str(row["genres"]).replace("|", " ")
            soup = f"{row['clean_title']} {genres_clean} {row['extracted_keywords']} {row['cast_names']} {row['director']} {row['tags_list']} {row['overview']}"
            soup = re.sub(r'\s+', ' ', soup).strip().lower()
            return soup
            
        self.df["soup"] = self.df.apply(create_soup, axis=1)
        self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])
        print("TF-IDF matrix built successfully.")
        
    def get_user_profile_preferences(self, user_id):
        genre_prefs = {}
        keyword_prefs = {}
        
        ratings = db.get_user_ratings(user_id)
        likes = db.get_user_likes_dislikes(user_id)
        watchlist = db.get_watchlist(user_id)
        history = db.get_watch_history(user_id)
        
        interacted_movies = {}
        
        def get_decay_weight(date_str):
            try:
                date_val = datetime.strptime(date_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                delta_days = (datetime.now() - date_val).days
                if delta_days <= 7:
                    return 1.0
                elif delta_days <= 30:
                    return 0.5
                else:
                    return 0.1
            except:
                return 0.2
                
        for h in history:
            m_id = h["movie_id"]
            decay = get_decay_weight(h["watched_at"])
            interacted_movies[m_id] = interacted_movies.get(m_id, 0.0) + (0.5 * decay)
            
        for w in watchlist:
            m_id = w["movie_id"]
            decay = get_decay_weight(w["added_at"])
            interacted_movies[m_id] = interacted_movies.get(m_id, 0.0) + (0.8 * decay)
            
        for m_id, r in ratings.items():
            r_impact = (r - 2.5) * 0.8
            interacted_movies[m_id] = interacted_movies.get(m_id, 0.0) + r_impact
            
        for m_id, is_like in likes.items():
            impact = 2.0 if is_like else -2.0
            interacted_movies[m_id] = interacted_movies.get(m_id, 0.0) + impact
            
        if not interacted_movies:
            return None, None
            
        for m_id, weight in interacted_movies.items():
            match = self.df[self.df["movieId"] == m_id]
            if len(match) > 0:
                row = match.iloc[0]
                genres_list = str(row["genres"]).split("|")
                for g in genres_list:
                    genre_prefs[g] = genre_prefs.get(g, 0.0) + weight
                    
                kws = str(row["extracted_keywords"]).split(",") + str(row["tags_list"]).split(",")
                for kw in kws:
                    kw_clean = kw.strip().lower()
                    if kw_clean and kw_clean != "nan" and kw_clean != "n/a":
                        keyword_prefs[kw_clean] = keyword_prefs.get(kw_clean, 0.0) + (weight * 0.2)
                        
        return genre_prefs, keyword_prefs
        
    def compute_profile_match_scores(self, user_id, candidates):
        genre_prefs, keyword_prefs = self.get_user_profile_preferences(user_id)
        if not genre_prefs and not keyword_prefs:
            return np.zeros(len(candidates))
            
        personal_scores = np.zeros(len(candidates))
        candidate_indices = candidates.index.tolist()
        
        for idx_pos, idx_val in enumerate(candidate_indices):
            row = candidates.loc[idx_val]
            score = 0.0
            genres_list = str(row["genres"]).split("|")
            for g in genres_list:
                score += genre_prefs.get(g, 0.0)
                
            keywords_text = str(row["extracted_keywords"]).lower() + " " + str(row["tags_list"]).lower()
            for kw, kw_weight in keyword_prefs.items():
                if kw in keywords_text:
                    score += kw_weight
                    
            personal_scores[idx_pos] = score
            
        s_min = personal_scores.min()
        s_max = personal_scores.max()
        if s_max > s_min:
            personal_scores = (personal_scores - s_min) / (s_max - s_min)
        else:
            personal_scores = np.zeros(len(candidates))
            
        return personal_scores

    def get_hybrid_recommendations(self, user_prompt_moods=None, target_movie_title=None, user_id=None, top_n=12, genre_filter=None, weights=None, conversational_query=None):
        """Generates movie recommendations blending mood, content TF-IDF matching, collaborative SVD, and user history decay."""
        start_t = time.time()
        
        # 1. Filter movies by genre if requested
        candidates = self.df.copy()
        if genre_filter:
            if isinstance(genre_filter, list):
                for g in genre_filter:
                    candidates = candidates[candidates["genres"].str.contains(g, na=False, case=False)]
            elif isinstance(genre_filter, str):
                candidates = candidates[candidates["genres"].str.contains(genre_filter, na=False, case=False)]
                
        if len(candidates) == 0:
            candidates = self.df.copy()
            
        # 2. Mood Matching Score
        mood_scores = np.zeros(len(candidates))
        if user_prompt_moods:
            moods = ["Happy", "Sad", "Excited", "Relaxed", "Inspirational", "Romantic", "Thoughtful", "Suspenseful"]
            input_vector = np.array([user_prompt_moods.get(m, 0.0) for m in moods])
            mood_columns = [f"mood_{m}" for m in moods]
            candidate_vectors = candidates[mood_columns].values
            
            input_norm = np.linalg.norm(input_vector)
            if input_norm > 0:
                candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
                candidate_norms = np.where(candidate_norms == 0, 1.0, candidate_norms)
                mood_scores = np.dot(candidate_vectors, input_vector) / (candidate_norms * input_norm)
            else:
                mood_scores = np.ones(len(candidates)) / len(candidates)
        else:
            mood_scores = np.ones(len(candidates))
            
        # 3. Conversational search or movie matching via high-performance TF-IDF cosine similarity
        content_scores = np.zeros(len(candidates))
        
        if conversational_query:
            # Match prompt text against movie metadata soups instantly via TF-IDF
            try:
                query_vector = self.vectorizer.transform([conversational_query.lower()])
                candidate_indices = candidates.index.tolist()
                candidate_vectors = self.tfidf_matrix[candidate_indices]
                
                sim = cosine_similarity(query_vector, candidate_vectors)
                content_scores = sim.flatten()
            except Exception as e:
                print(f"Error executing TF-IDF conversational search: {e}")
                content_scores = np.zeros(len(candidates))
                
        elif target_movie_title:
            matches = self.df[self.df["clean_title"].str.lower() == target_movie_title.lower()]
            if len(matches) == 0:
                matches = self.df[self.df["clean_title"].str.contains(target_movie_title, case=False, na=False)]
                
            if len(matches) > 0:
                target_idx = matches.index[0]
                target_vector = self.tfidf_matrix[target_idx]
                candidate_indices = candidates.index.tolist()
                candidate_vectors = self.tfidf_matrix[candidate_indices]
                
                sim = cosine_similarity(target_vector, candidate_vectors)
                content_scores = sim.flatten()
            else:
                content_scores = np.zeros(len(candidates))
        else:
            content_scores = np.zeros(len(candidates))
            
        # 4. Collaborative Filtering Score
        collab_scores = np.zeros(len(candidates))
        max_rating_count = self.df["rating_count"].max() if self.df["rating_count"].max() > 0 else 1.0
        popularity_scores = candidates["rating_count"].values / max_rating_count
        
        if self.svd_model and user_id is not None:
            candidate_movie_ids = candidates["movieId"].values
            preds = [self.svd_model.predict(user_id, int(m_id)).est for m_id in candidate_movie_ids]
            collab_scores = (np.array(preds) - 0.5) / 4.5
        else:
            avg_ratings = candidates["avg_rating"].fillna(2.5).values / 5.0
            collab_scores = avg_ratings * 0.7 + popularity_scores * 0.3
            
        collab_scores = np.clip(collab_scores, 0, 1)
        
        # 5. Personalized User Profile Score
        personal_scores = np.zeros(len(candidates))
        if user_id is not None:
            personal_scores = self.compute_profile_match_scores(user_id, candidates)
            
        # 6. Blending Weights
        w_mood = 0.40
        w_content = 0.30
        w_collab = 0.30
        w_personal = 0.00
        
        if weights is not None:
            w_mood = weights.get('mood', 0.40)
            w_content = weights.get('content', 0.30)
            w_collab = weights.get('collab', 0.30)
            w_personal = weights.get('personal', 0.00)
            
        if not target_movie_title and not conversational_query:
            w_mood += w_content * 0.6
            w_collab += w_content * 0.4
            w_content = 0.0
            
        if user_id is None or np.all(personal_scores == 0):
            w_mood += w_personal * 0.6
            w_collab += w_personal * 0.4
            w_personal = 0.0
            
        # Normalize weights
        total_w = w_mood + w_content + w_collab + w_personal
        if total_w > 0:
            w_mood /= total_w
            w_content /= total_w
            w_collab /= total_w
            w_personal /= total_w
            
        final_scores = (w_mood * mood_scores) + (w_content * content_scores) + (w_collab * collab_scores) + (w_personal * personal_scores)
        
        norm_avg_rating = candidates["avg_rating"].fillna(2.5).values / 5.0
        final_scores += 0.05 * (popularity_scores * norm_avg_rating)
        
        results = candidates.copy()
        results["mood_match_pct"] = mood_scores * 100
        results["content_match_pct"] = content_scores * 100
        results["collab_score_pct"] = collab_scores * 100
        results["personal_score_pct"] = personal_scores * 100
        results["match_score"] = final_scores
        
        if len(results) > top_n * 2:
            results = results[results["rating_count"] >= 5]
            
        top_results = results.sort_values(by="match_score", ascending=False).head(top_n)
        
        print(f"Recommendation blending completed in {time.time() - start_t:.3f} seconds.")
        return top_results
        
    def get_surprise_recommendation(self, user_id=None, user_prompt_moods=None, weights=None):
        candidates = self.df.copy()
        
        # Exclude watched and watchlist
        excluded_ids = set()
        if user_id is not None:
            history = db.get_watch_history(user_id)
            watchlist = db.get_watchlist(user_id)
            for h in history:
                excluded_ids.add(h["movie_id"])
            for w in watchlist:
                excluded_ids.add(w["movie_id"])
                
        if excluded_ids:
            candidates = candidates[~candidates["movieId"].isin(excluded_ids)]
            
        if len(candidates) == 0:
            candidates = self.df.copy()
            
        # Get baseline recommendations
        recs = self.get_hybrid_recommendations(
            user_prompt_moods=user_prompt_moods,
            user_id=user_id,
            top_n=50,
            weights=weights
        )
        
        if len(recs) == 0:
            recs = candidates.sort_values(by="popularity", ascending=False).head(50)
            recs["match_score"] = np.ones(len(recs)) * 0.5
            recs["mood_match_pct"] = 50.0
            recs["content_match_pct"] = 0.0
            recs["collab_score_pct"] = 50.0
            recs["personal_score_pct"] = 0.0
            
        # Add random noise
        noise = np.random.uniform(-0.15, 0.15, size=len(recs))
        recs["surprise_score"] = recs["match_score"] + noise
        
        recs = recs.sort_values(by="surprise_score", ascending=False)
        winner = recs.iloc[0]
        return winner
