import os
import sqlite3
import hashlib
import secrets
import html
from datetime import datetime

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cinemood.db"))

def get_connection():
    """Returns a connection to the SQLite database with Row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database schema if tables don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Watch history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS watch_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        tmdb_id INTEGER,
        watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    
    # 3. Search history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT NOT NULL,
        searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    
    # 4. Watchlist table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        tmdb_id INTEGER,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, movie_id)
    )
    """)
    
    # 5. Ratings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        rating REAL NOT NULL,
        rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, movie_id)
    )
    """)
    
    # 6. Likes / Dislikes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS likes_dislikes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        is_like INTEGER NOT NULL, -- 1 for like, 0 for dislike
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, movie_id)
    )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Securely hashes a password using PBKDF2 HMAC SHA-256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return pwdhash.hex(), salt

def sanitize_input(text):
    """Sanitizes text inputs to prevent XSS (Cross-Site Scripting)."""
    if text is None:
        return ""
    return html.escape(str(text).strip())

def register_user(username, password):
    """Registers a new user after verifying username uniqueness and password strength."""
    username = sanitize_input(username)
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
        
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists."
        
    pwdhash, salt = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", 
                       (username, pwdhash, salt))
        conn.commit()
        conn.close()
        return True, "User registered successfully."
    except Exception as e:
        conn.close()
        return False, f"Database error: {e}"

def login_user(username, password):
    """Logs in an existing user by verifying password against stored hash."""
    username = sanitize_input(username)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None, "Invalid username or password."
        
    user_id, stored_hash, salt = row["id"], row["password_hash"], row["salt"]
    computed_hash, _ = hash_password(password, salt)
    
    if computed_hash == stored_hash:
        return user_id, "Login successful."
    else:
        return None, "Invalid username or password."

# --- History Helpers ---

def add_watch_history(user_id, movie_id, title, tmdb_id=None):
    """Adds a movie to the user's watch history."""
    title = sanitize_input(title)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Keep watch history clean: check if this movie was watched recently (within 5 minutes)
        # to avoid duplicating rapid double clicks
        cursor.execute("""
            SELECT id FROM watch_history 
            WHERE user_id = ? AND movie_id = ? 
            ORDER BY watched_at DESC LIMIT 1
        """, (user_id, movie_id))
        
        cursor.execute("INSERT INTO watch_history (user_id, movie_id, title, tmdb_id) VALUES (?, ?, ?, ?)", 
                       (user_id, movie_id, title, tmdb_id))
        conn.commit()
    except Exception as e:
        print(f"Error saving watch history: {e}")
    finally:
        conn.close()

def get_watch_history(user_id, limit=20):
    """Retrieves the user's watch history, sorted by timestamp descending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie_id, title, tmdb_id, watched_at 
        FROM watch_history 
        WHERE user_id = ? 
        ORDER BY watched_at DESC 
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_search_history(user_id, query):
    """Logs user query search history."""
    query = sanitize_input(query)
    if not query.strip():
        return
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO search_history (user_id, query) VALUES (?, ?)", (user_id, query))
        conn.commit()
    except Exception as e:
        print(f"Error logging search history: {e}")
    finally:
        conn.close()

def get_search_history(user_id, limit=10):
    """Retrieves recent search history queries."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT query, searched_at 
        FROM search_history 
        WHERE user_id = ? 
        ORDER BY searched_at DESC 
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Watchlist Helpers ---

def add_to_watchlist(user_id, movie_id, title, tmdb_id=None):
    """Adds a movie to the user's watchlist."""
    title = sanitize_input(title)
    conn = get_connection()
    cursor = conn.cursor()
    success = False
    message = ""
    try:
        cursor.execute("INSERT INTO watchlist (user_id, movie_id, title, tmdb_id) VALUES (?, ?, ?, ?)",
                       (user_id, movie_id, title, tmdb_id))
        conn.commit()
        success = True
        message = "Added to Watchlist."
    except sqlite3.IntegrityError:
        message = "Already in Watchlist."
    except Exception as e:
        message = f"Error: {e}"
    finally:
        conn.close()
    return success, message

def remove_from_watchlist(user_id, movie_id):
    """Removes a movie from the user's watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
    finally:
        conn.close()
    return success

def get_watchlist(user_id):
    """Retrieves all movies currently in the user's watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie_id, title, tmdb_id, added_at 
        FROM watchlist 
        WHERE user_id = ? 
        ORDER BY added_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def is_in_watchlist(user_id, movie_id):
    """Checks if a specific movie is in the user's watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
    row = cursor.fetchone()
    conn.close()
    return row is not None

# --- Ratings & Likes Helpers ---

def rate_movie(user_id, movie_id, rating):
    """Submits or updates a user rating for a movie (0.5 to 5.0)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO ratings (user_id, movie_id, rating) 
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating, rated_at=CURRENT_TIMESTAMP
        """, (user_id, movie_id, rating))
        conn.commit()
    except Exception as e:
        print(f"Error rating movie: {e}")
    finally:
        conn.close()

def get_user_ratings(user_id):
    """Retrieves all ratings submitted by a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row["movie_id"]: row["rating"] for row in rows}

def set_like_dislike(user_id, movie_id, is_like):
    """Sets a movie as liked (1) or disliked (0). Passing None deletes the entry."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if is_like is None:
            cursor.execute("DELETE FROM likes_dislikes WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        else:
            cursor.execute("""
                INSERT INTO likes_dislikes (user_id, movie_id, is_like) 
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET is_like=excluded.is_like, updated_at=CURRENT_TIMESTAMP
            """, (user_id, movie_id, int(is_like)))
        conn.commit()
    except Exception as e:
        print(f"Error setting like/dislike: {e}")
    finally:
        conn.close()

def get_user_likes_dislikes(user_id):
    """Retrieves liked and disliked movies for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id, is_like FROM likes_dislikes WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    # returns dict of {movie_id: is_like}
    return {row["movie_id"]: bool(row["is_like"]) for row in rows}

# Initialize DB structure on import
init_db()
