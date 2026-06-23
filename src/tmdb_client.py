import os
import requests
from functools import lru_cache
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class TMDBClient:
    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY", "")
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
        # Premium placeholder image from Unsplash for missing posters
        self.placeholder_poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"
        
        if self.api_key:
            print("TMDB Client initialized with API Key.")
        else:
            print("TMDB Client initialized WITHOUT API Key. Using local metadata and search engine fallbacks.")

    def set_api_key(self, api_key):
        """Allows dynamic setting of API key in the app interface"""
        self.api_key = api_key
        
    @lru_cache(maxsize=1024)
    def get_movie_details(self, tmdb_id, title, local_poster_path=""):
        """
        Fetches movie poster and trailer URL.
        Falls back to local poster path or placeholder, and YouTube search link for trailers.
        """
        poster_url = self.placeholder_poster
        trailer_url = ""
        
        # 1. Clean title for YouTube searches
        clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()
        search_query = f"{clean_title.replace(' ', '+')}+official+trailer"
        fallback_trailer = f"https://www.youtube.com/results?search_query={search_query}"
        
        # 2. Setup poster from local metadata if available
        if local_poster_path and isinstance(local_poster_path, str) and local_poster_path.strip() != "":
            # Format path correctly
            path = local_poster_path.strip()
            if not path.startswith("/"):
                path = "/" + path
            poster_url = f"{self.image_base_url}{path}"
            
        # 3. Call TMDB API if key is available
        if self.api_key and tmdb_id and not pandas_isna(tmdb_id):
            try:
                # Convert tmdb_id to integer
                t_id = int(float(tmdb_id))
                url = f"{self.base_url}/movie/{t_id}"
                params = {
                    "api_key": self.api_key,
                    "append_to_response": "videos"
                }
                
                response = requests.get(url, params=params, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update poster URL if TMDB returns one
                    if data.get("poster_path"):
                        poster_url = f"{self.image_base_url}{data['poster_path']}"
                        
                    # Find YouTube trailer
                    videos = data.get("videos", {}).get("results", [])
                    trailer_key = None
                    # Search for trailer
                    for vid in videos:
                        if vid.get("site") == "YouTube" and vid.get("type") == "Trailer":
                            trailer_key = vid.get("key")
                            break
                    # Fallback to clip/teaser if no official trailer
                    if not trailer_key and videos:
                        for vid in videos:
                            if vid.get("site") == "YouTube":
                                trailer_key = vid.get("key")
                                break
                                
                    if trailer_key:
                        trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
            except Exception as e:
                # Silently fallback on timeout or connection error
                pass
                
        # If no API key trailer found, use YouTube search page as fallback
        if not trailer_url:
            trailer_url = fallback_trailer
            
        return {
            "poster_url": poster_url,
            "trailer_url": trailer_url
        }

def pandas_isna(val):
    # Safe NaN checker without importing pandas globally
    try:
        import pandas as pd
        return pd.isna(val)
    except:
        import numpy as np
        try:
            return np.isnan(val)
        except:
            return val is None or val != val

if __name__ == "__main__":
    # Test client
    client = TMDBClient()
    # Test Toy Story (tmdbId = 862)
    details = client.get_movie_details(862, "Toy Story (1995)", "/uXDfjJbdP4m6jGVwH5IEzwi4n15.jpg")
    print("Poster URL:", details["poster_url"])
    print("Trailer URL:", details["trailer_url"])
