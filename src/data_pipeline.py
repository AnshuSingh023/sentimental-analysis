import os
import pandas as pd
import numpy as np
import ast
import time

def parse_json_col(val, key='name', limit=None):
    if pd.isna(val) or not isinstance(val, str):
        return ""
    try:
        # Using ast.literal_eval is safer than json.loads for standard single-quoted CSV lists
        data = ast.literal_eval(val)
        if not isinstance(data, list):
            return ""
        names = [d[key] for d in data if isinstance(d, dict) and key in d]
        if limit:
            names = names[:limit]
        return ", ".join(names)
    except:
        try:
            # Fallback to json if literal_eval fails
            import json
            data = json.loads(val)
            if not isinstance(data, list):
                return ""
            names = [d[key] for d in data if isinstance(d, dict) and key in d]
            if limit:
                names = names[:limit]
            return ", ".join(names)
        except:
            return ""

def get_director(val):
    if pd.isna(val) or not isinstance(val, str):
        return ""
    try:
        data = ast.literal_eval(val)
        if not isinstance(data, list):
            return ""
        for d in data:
            if isinstance(d, dict) and d.get('job') == 'Director':
                return d.get('name', '')
        return ""
    except:
        try:
            import json
            data = json.loads(val)
            if not isinstance(data, list):
                return ""
            for d in data:
                if isinstance(d, dict) and d.get('job') == 'Director':
                    return d.get('name', '')
            return ""
        except:
            return ""

def run_pipeline():
    start_time = time.time()
    print("Starting CineMood AI Data Pipeline...")
    
    # Define file paths
    movielens_dir = "data/raw/movielens"
    tmdb_dir = "data/tmdb"
    output_dir = "data"
    
    # Check that input files exist
    required_files = [
        os.path.join(movielens_dir, "movies.csv"),
        os.path.join(movielens_dir, "ratings.csv"),
        os.path.join(movielens_dir, "tags.csv"),
        os.path.join(movielens_dir, "links.csv"),
        os.path.join(tmdb_dir, "movies_metadata.csv"),
        os.path.join(tmdb_dir, "keywords.csv"),
        os.path.join(tmdb_dir, "credits.csv")
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")
            
    # 1. Process MovieLens Ratings (Memory-efficient aggregation)
    print("Step 1: Loading and aggregating MovieLens ratings (32M rows)...")
    ratings = pd.read_csv(
        os.path.join(movielens_dir, "ratings.csv"),
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"}
    )
    print(f"Loaded {len(ratings):,} ratings. Grouping by movieId...")
    movie_ratings = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    del ratings # Free memory
    print(f"Aggregated ratings for {len(movie_ratings):,} unique movies.")
    
    # 2. Process MovieLens Tags
    print("Step 2: Loading and aggregating MovieLens tags (2M rows)...")
    tags = pd.read_csv(
        os.path.join(movielens_dir, "tags.csv"),
        usecols=["movieId", "tag"],
        dtype={"movieId": "int32", "tag": "str"}
    )
    print(f"Loaded {len(tags):,} tags. Grouping by movieId...")
    # Fill NA tags and clean
    tags["tag"] = tags["tag"].fillna("").astype(str).str.strip().str.lower()
    movie_tags = tags.groupby("movieId").agg(
        tag_count=("tag", "count"),
        tags_list=("tag", lambda x: " | ".join(list(set(filter(None, x)))[:20])) # Limit to top 20 unique tags to save space
    ).reset_index()
    del tags # Free memory
    print(f"Aggregated tags for {len(movie_tags):,} unique movies.")
    
    # 3. Load MovieLens movies & links
    print("Step 3: Loading MovieLens movies and links...")
    movies = pd.read_csv(os.path.join(movielens_dir, "movies.csv"))
    links = pd.read_csv(os.path.join(movielens_dir, "links.csv"))
    
    # Convert link IDs to numeric (coercing TMDB IDs since some are missing)
    links["movieId"] = links["movieId"].astype(int)
    links["imdbId"] = links["imdbId"].astype(int)
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    
    # Merge MovieLens files
    print("Merging MovieLens metadata...")
    ml_master = movies.merge(movie_ratings, on="movieId", how="left")
    ml_master = ml_master.merge(movie_tags, on="movieId", how="left")
    ml_master = ml_master.merge(links, on="movieId", how="left")
    
    # Fill NaNs for counts
    ml_master["rating_count"] = ml_master["rating_count"].fillna(0).astype(int)
    ml_master["tag_count"] = ml_master["tag_count"].fillna(0).astype(int)
    ml_master["tags_list"] = ml_master["tags_list"].fillna("")
    
    print(f"MovieLens master size: {len(ml_master):,}")
    
    # 4. Load TMDB Metadata, Keywords, and Credits
    print("Step 4: Loading and cleaning TMDB movies metadata...")
    tmdb_meta = pd.read_csv(
        os.path.join(tmdb_dir, "movies_metadata.csv"),
        usecols=["id", "overview", "popularity", "poster_path", "vote_average", "vote_count", "release_date", "tagline"],
        low_memory=False
    )
    
    # Clean ID column (some rows are corrupted with text or release dates in id field)
    tmdb_meta["id"] = pd.to_numeric(tmdb_meta["id"], errors="coerce")
    tmdb_meta = tmdb_meta.dropna(subset=["id"]).copy()
    tmdb_meta["id"] = tmdb_meta["id"].astype(int)
    tmdb_meta = tmdb_meta.drop_duplicates(subset=["id"])
    
    # Load keywords
    print("Loading TMDB keywords...")
    tmdb_keywords = pd.read_csv(os.path.join(tmdb_dir, "keywords.csv"))
    tmdb_keywords["id"] = pd.to_numeric(tmdb_keywords["id"], errors="coerce")
    tmdb_keywords = tmdb_keywords.dropna(subset=["id"]).copy()
    tmdb_keywords["id"] = tmdb_keywords["id"].astype(int)
    tmdb_keywords = tmdb_keywords.drop_duplicates(subset=["id"])
    
    # Load credits
    print("Loading TMDB credits...")
    tmdb_credits = pd.read_csv(os.path.join(tmdb_dir, "credits.csv"))
    tmdb_credits["id"] = pd.to_numeric(tmdb_credits["id"], errors="coerce")
    tmdb_credits = tmdb_credits.dropna(subset=["id"]).copy()
    tmdb_credits["id"] = tmdb_credits["id"].astype(int)
    tmdb_credits = tmdb_credits.drop_duplicates(subset=["id"])
    
    # 5. Extract features from TMDB
    print("Step 5: Extracting features from TMDB JSON structure...")
    
    # Keywords extraction
    print("Extracting keywords...")
    tmdb_keywords["extracted_keywords"] = tmdb_keywords["keywords"].apply(lambda x: parse_json_col(x, 'name', limit=15))
    tmdb_keywords = tmdb_keywords[["id", "extracted_keywords"]]
    
    # Credits extraction (top 4 cast and director)
    print("Extracting cast and director...")
    tmdb_credits["cast_names"] = tmdb_credits["cast"].apply(lambda x: parse_json_col(x, 'name', limit=4))
    tmdb_credits["director"] = tmdb_credits["crew"].apply(get_director)
    tmdb_credits = tmdb_credits[["id", "cast_names", "director"]]
    
    # Merge TMDB sub-datasets
    print("Merging TMDB components...")
    tmdb_full = tmdb_meta.merge(tmdb_keywords, on="id", how="left")
    tmdb_full = tmdb_full.merge(tmdb_credits, on="id", how="left")
    
    # Rename TMDB ID to tmdbId to merge with MovieLens
    tmdb_full = tmdb_full.rename(columns={"id": "tmdbId"})
    print(f"TMDB unified metadata size: {len(tmdb_full):,}")
    
    # 6. Merge MovieLens & TMDB
    print("Step 6: Merging MovieLens knowledge base with TMDB features...")
    # Join on tmdbId
    master_df = ml_master.merge(tmdb_full, on="tmdbId", how="left")
    
    # Fill remaining NaNs with appropriate placeholders
    master_df["overview"] = master_df["overview"].fillna("")
    master_df["tagline"] = master_df["tagline"].fillna("")
    master_df["extracted_keywords"] = master_df["extracted_keywords"].fillna("")
    master_df["cast_names"] = master_df["cast_names"].fillna("")
    master_df["director"] = master_df["director"].fillna("")
    master_df["poster_path"] = master_df["poster_path"].fillna("")
    master_df["popularity"] = pd.to_numeric(master_df["popularity"], errors="coerce").fillna(0.0)
    master_df["vote_average"] = pd.to_numeric(master_df["vote_average"], errors="coerce").fillna(0.0)
    master_df["vote_count"] = pd.to_numeric(master_df["vote_count"], errors="coerce").fillna(0).astype(int)
    
    # Extract release year from release_date or fallback to MovieLens title year extraction
    print("Extracting release years...")
    master_df["release_year"] = pd.to_datetime(master_df["release_date"], errors="coerce").dt.year
    
    # Fallback title parsing for year e.g. Toy Story (1995) -> 1995
    def extract_year_from_title(title):
        import re
        match = re.search(r'\((\d{4})\)', str(title))
        if match:
            return int(match.group(1))
        return np.nan
        
    title_years = master_df["title"].apply(extract_year_from_title)
    master_df["release_year"] = master_df["release_year"].fillna(title_years).fillna(0).astype(int)
    
    # Clean titles: remove trailing spaces and year brackets from title if desired, but let's keep clean_title separate
    master_df["clean_title"] = master_df["title"].apply(lambda x: re.sub(r'\s*\(\d{4}\)\s*$', '', str(x)).strip() if isinstance(x, str) else "")
    
    # 7. Write to Master CSV
    print(f"Step 7: Saving master movies dataset to {os.path.join(output_dir, 'master_movies.csv')}...")
    master_df.to_csv(os.path.join(output_dir, "master_movies.csv"), index=False)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Data Pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f"Final master_movies shape: {master_df.shape}")

if __name__ == "__main__":
    import re
    run_pipeline()
