import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure database to use a test file
import src.database as db
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_cinemood.db"))
db.DB_PATH = TEST_DB_PATH

from src.sentiment_model import SentimentPredictor
from src.emotion_model import CineMoodEmotionDetector
from src.recommender_engine import CineMoodRecommender
from src.tmdb_client import TMDBClient

class TestCineMoodPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Initialize test database
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        db.init_db()
        
    @classmethod
    def tearDownClass(cls):
        # Cleanup test database
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
            
    def test_01_master_dataset(self):
        """Test if the master dataset is created and has correct columns"""
        data_path = "data/master_movies.csv"
        self.assertTrue(os.path.exists(data_path), f"Master dataset not found at {data_path}")
        
        df = pd.read_csv(data_path, nrows=50)
        required_cols = [
            "movieId", "title", "genres", "avg_rating", "rating_count", 
            "tag_count", "imdbId", "tmdbId", "overview", "popularity", 
            "poster_path", "vote_average", "vote_count", "release_year"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Required column {col} missing from master movies.")
            
        print("Dataset verification passed.")
        
    def test_02_database_operations(self):
        """Test secure authentication and history tracking in SQLite database"""
        # Test Registration
        reg_ok, msg = db.register_user("testuser", "securepass123")
        self.assertTrue(reg_ok)
        
        # Test Duplicate Registration
        reg_dup, msg_dup = db.register_user("testuser", "securepass123")
        self.assertFalse(reg_dup)
        
        # Test Login
        uid, msg_login = db.login_user("testuser", "securepass123")
        self.assertIsNotNone(uid)
        
        # Test Wrong Password Login
        uid_wrong, msg_wrong = db.login_user("testuser", "wrongpassword")
        self.assertIsNone(uid_wrong)
        
        # Test Watchlist
        db.add_to_watchlist(uid, 1, "Toy Story", 862)
        self.assertTrue(db.is_in_watchlist(uid, 1))
        
        wl = db.get_watchlist(uid)
        self.assertEqual(len(wl), 1)
        self.assertEqual(wl[0]["title"], "Toy Story")
        
        # Test History
        db.add_watch_history(uid, 1, "Toy Story", 862)
        hist = db.get_watch_history(uid)
        self.assertEqual(len(hist), 1)
        
        print("Database/Security SQL verification passed.")
        
    def test_03_emotion_detector(self):
        """Test the mapping from text to 25 movie emotions & crisis filters"""
        detector = CineMoodEmotionDetector(use_transformer=False) # Use lexical fallback for testing speed
        
        # Test happy prompt
        happy_res = detector.get_mood_scores("I want a happy and funny comedy to lift my spirits.")
        self.assertEqual(happy_res["dominant_mood"], "Happy")
        self.assertIn("Happy", happy_res["mood_scores"])
        self.assertFalse(happy_res["is_crisis"])
        
        # Test crisis self-harm filter
        crisis_res = detector.get_mood_scores("I want to kill myself today")
        self.assertTrue(crisis_res["is_crisis"])
        self.assertEqual(crisis_res["dominant_mood"], "Sad")
        
        print("Emotion & Safety Crisis verification passed.")
        
    def test_04_recommender_hybrid(self):
        """Test hybrid blending recommendations"""
        recommender = CineMoodRecommender(use_semantic=False)
        
        dummy_moods = {
            "Happy": 0.6,
            "Relaxed": 0.4
        }
        
        recs = recommender.get_hybrid_recommendations(
            user_prompt_moods=dummy_moods,
            top_n=5
        )
        
        self.assertEqual(len(recs), 5, f"Expected 5 recommendations, got {len(recs)}")
        self.assertIn("match_score", recs.columns)
        self.assertIn("avg_rating", recs.columns)
        
        print("Hybrid Recommender Engine verification passed.")
        
    def test_05_tmdb_client(self):
        """Test TMDB Client image formatting and fallback YouTube links"""
        client = TMDBClient()
        details = client.get_movie_details(tmdb_id=862, title="Toy Story", local_poster_path="/toy_story.jpg")
        
        self.assertEqual(details["poster_url"], "https://image.tmdb.org/t/p/w500/toy_story.jpg")
        self.assertIn("youtube.com", details["trailer_url"])
        
        print("TMDB Real-Time Integration verification passed.")

    def test_06_custom_weights(self):
        """Test if the hybrid recommender respects custom blend weights"""
        recommender = CineMoodRecommender(use_semantic=False)
        
        dummy_moods = {
            "Happy": 0.8,
            "Relaxed": 0.2
        }
        
        recs_mood_only = recommender.get_hybrid_recommendations(
            user_prompt_moods=dummy_moods,
            top_n=5,
            weights={'mood': 1.0, 'content': 0.0, 'collab': 0.0, 'personal': 0.0}
        )
        
        recs_collab_only = recommender.get_hybrid_recommendations(
            user_prompt_moods=dummy_moods,
            top_n=5,
            weights={'mood': 0.0, 'content': 0.0, 'collab': 1.0, 'personal': 0.0}
        )
        
        self.assertFalse(recs_mood_only["movieId"].equals(recs_collab_only["movieId"]), 
                         "Recommendations for 100% mood vs 100% collab weights should differ.")
        
        print("Custom Blend Weights verification passed.")

if __name__ == "__main__":
    unittest.main()
