import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import html

# Ensure root directory is on path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_model import SentimentPredictor
from src.emotion_model import CineMoodEmotionDetector
from src.recommender_engine import CineMoodRecommender
from src.tmdb_client import TMDBClient
import src.database as db

genres_all = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Set up Streamlit Page config
st.set_page_config(
    page_title="CineMood AI - Emotion-Aware Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS styling
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize models and wrap in cache resource (runs in lightweight lexical mode now)
@st.cache_resource(show_spinner=True)
def get_models():
    sentiment_pred = None
    try:
        sentiment_pred = SentimentPredictor()
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        
    emotion_det = CineMoodEmotionDetector(use_transformer=False) # Lightweight lexical classifier
    recommender = CineMoodRecommender(use_semantic=False) # Fast TF-IDF prompt search
    tmdb = TMDBClient()
    return sentiment_pred, emotion_det, recommender, tmdb

sentiment_model, emotion_model, recommender_model, tmdb_client = get_models()

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "skipped_login" not in st.session_state:
    st.session_state.skipped_login = False
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "active_preset" not in st.session_state:
    st.session_state.active_preset = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "recommendations_type" not in st.session_state:
    st.session_state.recommendations_type = None # 'standard' or 'surprise'
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False
if "tmdb_api_key" not in st.session_state:
    st.session_state.tmdb_api_key = os.getenv("TMDB_API_KEY", "")
    tmdb_client.set_api_key(st.session_state.tmdb_api_key)

# Clean HTML renderer helper to prevent leading whitespaces from turning into pre code blocks
def render_movie_card(movie, details):
    if details["poster_url"] == tmdb_client.placeholder_poster or not details["poster_url"]:
        poster_html = f"<div class='movie-poster-container'><div class='poster-fallback'><div class='poster-fallback-watermark'>🎬 CineMood AI</div><div class='poster-fallback-not-available'>Poster Not Available</div><div class='poster-fallback-title'>{html.escape(movie['clean_title'])}</div><div class='poster-fallback-meta'><span>📅 {movie['release_year']}</span><span>🎭 {html.escape(str(movie['genres']).split('|')[0])}</span></div></div></div>"
    else:
        poster_html = f"<div class='movie-poster-container'><img src='{details['poster_url']}' class='movie-poster'></div>"
        
    card_html = f"<div class='movie-card'>{poster_html}<div class='movie-info'><div class='movie-title' title='{html.escape(movie['clean_title'])}'>{html.escape(movie['clean_title'])} ({movie['release_year']})</div><div class='movie-meta'><span class='match-badge'>{movie['match_score']*100:.0f}% Match</span><span class='rating-badge'>⭐ {movie['avg_rating']:.1f}</span></div><p style='font-size:0.72rem; color:#94a3b8; margin: 6px 0 0 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{html.escape(str(movie['genres']).replace('|', ' • '))}</p></div></div>"
    st.markdown(card_html, unsafe_allow_html=True)

# ----------------- FIRST PAGE LOCK SCREEN: LOGIN / REGISTER -----------------
if st.session_state.user_id is None and not st.session_state.skipped_login:
    st.markdown("<div class='gradient-text'>CineMood AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>An Emotion-Aware Movie Recommendation System powered by Sentiment Analysis, NLP, and Collaborative Filtering</div>", unsafe_allow_html=True)
    
    with st.container(key="gateway_container", border=True):
        st.markdown("<h3 style='text-align: center; color:#ffd700; margin-top:0;'>🍿 CineMood Gateway</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Log in to unlock personalized history tracking, custom watchlists, and rating preference learning.</p>", unsafe_allow_html=True)
        
        auth_mode = st.radio("Choose Mode", ["Log In", "Register Account"], horizontal=True)
        
        u_name = st.text_input("Username", key="gateway_username")
        p_word = st.text_input("Password", type="password", key="gateway_password")
        
        col_auth_btns = st.columns([1, 1])
        
        with col_auth_btns[0]:
            submit_auth = st.button("Confirm 🔓", use_container_width=True, key="gateway_confirm_btn")
            
        with col_auth_btns[1]:
            skip_auth = st.button("Explore as Guest 🧑‍🚀", use_container_width=True, key="gateway_skip_btn")
            
        if submit_auth:
            if auth_mode == "Register Account":
                success, msg = db.register_user(u_name, p_word)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                uid, msg = db.login_user(u_name, p_word)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = db.sanitize_input(u_name)
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)
                    
        if skip_auth:
            st.session_state.skipped_login = True
            st.rerun()
    
else:
    # ----------------- MAIN APP SETUP -----------------
    
    # Sidebar setup
    st.sidebar.markdown("<h2 style='text-align: center;'>🔐 CineMood Auth</h2>", unsafe_allow_html=True)
    
    if st.session_state.user_id is None:
        st.sidebar.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>You are exploring as a guest.</p>", unsafe_allow_html=True)
        auth_mode = st.sidebar.radio("Sign In / Sign Up", ["Login", "Register"], key="sidebar_auth_mode")
        
        with st.sidebar.form("sidebar_auth_form"):
            u_name = st.text_input("Username")
            p_word = st.text_input("Password", type="password")
            submit_auth = st.form_submit_button(auth_mode)
            
            if submit_auth:
                if auth_mode == "Register":
                    success, msg = db.register_user(u_name, p_word)
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)
                else:
                    uid, msg = db.login_user(u_name, p_word)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = db.sanitize_input(u_name)
                        st.sidebar.success(msg)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.sidebar.error(msg)
    else:
        st.sidebar.markdown(f"<div style='text-align:center; padding: 10px; border-radius:10px; background:rgba(255,46,147,0.1); border:1px solid rgba(255,46,147,0.2);'>Welcome, <strong>{st.session_state.username}</strong>! 👋</div>", unsafe_allow_html=True)
        if st.sidebar.button("Logout 🚪", use_container_width=True, key="sidebar_logout_btn"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.skipped_login = False
            st.sidebar.info("Logged out successfully.")
            time.sleep(0.5)
            st.rerun()
            
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Tuning Settings</h2>", unsafe_allow_html=True)
    
    # Settings Preset Selection
    preset_settings = st.sidebar.selectbox(
        "Choose a Tuning Preset:",
        ["Balanced", "Mood Focused", "Personalized", "Discovery", "Trending"],
        help="Balanced splits weights. Mood Focused prioritizes feelings. Personalized weights history highly."
    )
    
    if preset_settings == "Balanced":
        w_mood_default, w_content_default, w_collab_default, w_personal_default = 0.35, 0.25, 0.25, 0.15
    elif preset_settings == "Mood Focused":
        w_mood_default, w_content_default, w_collab_default, w_personal_default = 0.70, 0.10, 0.10, 0.10
    elif preset_settings == "Personalized":
        w_mood_default, w_content_default, w_collab_default, w_personal_default = 0.10, 0.10, 0.40, 0.40
    elif preset_settings == "Discovery":
        w_mood_default, w_content_default, w_collab_default, w_personal_default = 0.20, 0.30, 0.20, 0.30
    else: # Trending
        w_mood_default, w_content_default, w_collab_default, w_personal_default = 0.20, 0.20, 0.50, 0.10
        
    w_mood = st.sidebar.slider("Mood Match Weight", 0.0, 1.0, w_mood_default, 0.05)
    w_content = st.sidebar.slider("Content Similarity Weight", 0.0, 1.0, w_content_default, 0.05)
    w_collab = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, w_collab_default, 0.05)
    w_personal = st.sidebar.slider("Personal Profile Weight", 0.0, 1.0, w_personal_default, 0.05)
    
    sum_w = w_mood + w_content + w_collab + w_personal
    if sum_w > 0:
        w_mood_n = w_mood / sum_w
        w_content_n = w_content / sum_w
        w_collab_n = w_collab / sum_w
        w_personal_n = w_personal / sum_w
    else:
        w_mood_n, w_content_n, w_collab_n, w_personal_n = 0.4, 0.2, 0.2, 0.2
        
    st.sidebar.caption(f"Normalized Weights: Mood ({w_mood_n:.0%}), Content ({w_content_n:.0%}), Collab ({w_collab_n:.0%}), Personal ({w_personal_n:.0%})")
    
    # Dynamic TMDB API Key update
    api_key_input = st.sidebar.text_input(
        "TMDB API Key (Optional)", 
        value=st.session_state.tmdb_api_key, 
        type="password",
        help="Provide a TMDB API Key for live trailer embeds. If empty, local fallbacks are used.",
        key="sidebar_tmdb_key_input"
    )
    if api_key_input != st.session_state.tmdb_api_key:
        st.session_state.tmdb_api_key = api_key_input
        tmdb_client.set_api_key(api_key_input)
        st.sidebar.success("TMDB API Key updated!")
        
    # Navigation tabs (Ask Anything Mode combined into a single unified search tab for efficiency)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 AI recommendation Engine", 
        "💖 My Watchlist & Profile",
        "📊 Analytics Dashboard", 
        "📄 About CineMood"
    ])
    
    presets = {
        "🌟 Uplifting & Joyful": "I had a long, tiring day. Show me something funny, inspiring, and happy that will make me smile.",
        "🕯️ Romantic & Cozy": "Want a cozy, romantic love story with a warm feel.",
        "😱 Thrilling & Suspense": "I want an intense, scary mystery thriller with plenty of suspense and plot twists.",
        "🧠 Deep & Thoughtful": "Looking for a philosophical, deep, mind-bending sci-fi movie that makes me think.",
        "🌧️ Melancholic & Comfort": "I am feeling sad and down, and want a comforting drama to suit my mood."
    }
    
    # ----------------- TAB 1: Unified Recommender & Prompt Search -----------------
    with tab1:
        col_input, col_stats = st.columns([13, 7])
        with col_input:
            with st.container(border=True):
                st.markdown("### How are you feeling or what story do you want?")
                
                # Setup preset text values in session state
                if "mood_text_area" not in st.session_state:
                    st.session_state.mood_text_area = ""
                    
                # User text prompt input
                user_input = st.text_area(
                    "Describe your current mood or tell me what story you want to experience:",
                    height=110,
                    key="mood_text_area",
                    placeholder="e.g. I am feeling stressed out and want an uplifting comedy... OR a space adventure like Interstellar..."
                )
                
                st.session_state.user_prompt = user_input
                
                # Action to run presets
                def select_preset(prompt_val, name):
                    st.session_state.mood_text_area = prompt_val
                    st.session_state.active_preset = name
                    st.session_state.user_prompt = prompt_val
                    st.session_state.preset_just_clicked = True
                    
                # Render presets
                st.caption("Or pick a quick mood preset:")
                with st.container(key="presets_container"):
                    for i, (name, prompt_val) in enumerate(presets.items()):
                        is_active = st.session_state.active_preset == name
                        label = f"🔥 {name}" if is_active else name
                        
                        st.button(
                            label, 
                            key=f"preset_{i}", 
                            on_click=select_preset, 
                            args=(prompt_val, name)
                        )
                    
                # Optional genre filtering
                selected_genres = st.multiselect("Filter by Genres (Optional)", options=genres_all, key="recs_genre_filter")
                
                # Submit buttons
                col_buttons = st.columns([3, 2])
                with col_buttons[0]:
                    recommend_btn = st.button("Generate Recommendations 🚀", use_container_width=True, key="generate_recs_btn")
                with col_buttons[1]:
                    surprise_btn = st.button("🎲 Surprise Me!", use_container_width=True, key="surprise_recs_btn")
            
        # Variables to hold classification results
        user_moods = None
        sentiment_res = None
        is_crisis = False
        
        if st.session_state.user_prompt:
            res = emotion_model.get_mood_scores(st.session_state.user_prompt)
            user_moods = res["mood_scores"]
            dominant_mood = res["dominant_mood"]
            is_crisis = res["is_crisis"]
            
            if sentiment_model:
                sentiment_res = sentiment_model.predict(st.session_state.user_prompt)
                
        with col_stats:
            if st.session_state.user_prompt and user_moods:
                with st.container(border=True):
                    st.markdown("### Detected Emotion Profile")
                    
                    if is_crisis:
                        st.markdown("<div class='badge-dominant' style='background: #e50914;'>⚠️ SAFETY TRIGGER</div>", unsafe_allow_html=True)
                    else:
                        prim_badge = f"<span class='badge-dominant'>🎭 {res['primary_emotion']}</span>"
                        sec_badge = f"<span class='badge-secondary'>🎭 {res['secondary_emotion']}</span>"
                        st.markdown(f"Emotions: {prim_badge} {sec_badge}", unsafe_allow_html=True)
                        
                        # Strategy Descriptors
                        rec_strategy = "Standard Blend"
                        if res['primary_emotion'] in ["Sad", "Heartbroken", "Grieving", "Hopeless", "Depressed", "Melancholic"]:
                            rec_strategy = "Emotional Release (Cathartic Drama)"
                        elif res['primary_emotion'] in ["Stressed", "Anxious", "Lonely"]:
                            rec_strategy = "Uplifting Comfort & Escapism"
                        elif res['primary_emotion'] in ["Calm", "Relaxed"]:
                            rec_strategy = "Chill Ambient Flow"
                        elif res['primary_emotion'] in ["Inspired", "Motivated"]:
                            rec_strategy = "Uplifting Triumph Journey"
                        elif res['primary_emotion'] in ["Excited", "Adventurous"]:
                            rec_strategy = "High Adrenaline Action"
                        elif res['primary_emotion'] in ["Thoughtful", "Curious"]:
                            rec_strategy = "Intellectual/Mind-Bending Journey"
                            
                        st.markdown(f"<p style='margin: 8px 0; font-size:0.95rem; color:#ffd700;'>Strategy: <strong>{rec_strategy}</strong></p>", unsafe_allow_html=True)
                        
                    if sentiment_res:
                        score = sentiment_res["score"]
                        sent_label = sentiment_res["sentiment"]
                        sent_color = "#ff2e93" if sent_label == "positive" else "#e50914"
                        
                        # Plotly Sentiment indicator gauge (Pink-Red-Yellow steps)
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                                'bar': {'color': sent_color},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 1,
                                'bordercolor': "rgba(255,255,255,0.1)",
                                'steps': [
                                    {'range': [-1, -0.3], 'color': 'rgba(229, 9, 20, 0.1)'},
                                    {'range': [-0.3, 0.3], 'color': 'rgba(255, 215, 0, 0.1)'},
                                    {'range': [0.3, 1.0], 'color': 'rgba(255, 46, 147, 0.1)'}
                                ]
                            }
                        ))
                        fig_gauge.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0", size=10),
                            height=125,
                            margin=dict(l=20, r=20, t=10, b=10)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                with st.container(border=True):
                    st.markdown("<div style='text-align: center; color: #94a3b8; padding: 40px 0;'>Describe your mood or choose a preset to check recommendation strategy metrics.</div>", unsafe_allow_html=True)
                
        # 1. Trigger Recommendation calculations on button clicks or preset clicks
        preset_clicked_flag = st.session_state.get("preset_just_clicked", False)
        if (recommend_btn or preset_clicked_flag) and st.session_state.user_prompt:
            st.session_state.preset_just_clicked = False
            with st.spinner("Processing matching indices..."):
                recs_df = recommender_model.get_hybrid_recommendations(
                    user_prompt_moods=user_moods,
                    conversational_query=st.session_state.user_prompt,
                    user_id=st.session_state.user_id,
                    top_n=12,
                    genre_filter=selected_genres if len(selected_genres) > 0 else None,
                    weights={'mood': w_mood_n, 'content': w_content_n, 'collab': w_collab_n, 'personal': w_personal_n}
                )
                st.session_state.recommendations = recs_df.to_dict('records')
                st.session_state.recommendations_type = 'standard'
                st.session_state.show_recs = True
                
        elif surprise_btn:
            with st.spinner("Rolling mood roulette..."):
                surprise_movie = recommender_model.get_surprise_recommendation(
                    user_id=st.session_state.user_id,
                    user_prompt_moods=user_moods,
                    weights={'mood': w_mood_n, 'content': w_content_n, 'collab': w_collab_n, 'personal': w_personal_n}
                )
                st.session_state.recommendations = [surprise_movie.to_dict()]
                st.session_state.recommendations_type = 'surprise'
                st.session_state.show_recs = True
                
        # 2. Safety self-harm warning output
        if st.session_state.user_prompt and is_crisis:
            st.session_state.show_recs = False
            st.markdown(f"""
            <div class='crisis-box'>
                <div class='crisis-title'>❤️ We Are Here For You</div>
                <p style='color: #ffffff; font-size: 1.1rem; line-height: 1.6; margin-top:0;'>
                    It sounds like you are going through a difficult time. Please remember that you don't have to carry this alone. There are people who care and want to support you.
                </p>
                <p style='color: #ffd700; font-weight: 700; font-size: 1.15rem; margin-top: 15px; margin-bottom: 15px;'>
                    📞 Suicide & Crisis Lifeline: Call or text 988 (Available 24/7, Free, Confidential)
                </p>
                <p style='color: #e2e8f0; font-size: 0.95rem; margin-bottom:15px;'>
                    If you are outside the United States, please click below to find resources in your country.
                </p>
                <a href='https://findahelpline.com/' target='_blank' style='display:inline-block; background:#e50914; color:white; padding:10px 24px; border-radius:8px; text-decoration:none; font-weight:bold; box-shadow:0 4px 15px rgba(229,9,20,0.3);'>Find International Support Lines</a>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Render recommendations from Session State Cache (prevents reset on click actions!)
        if st.session_state.show_recs and not is_crisis:
            st.markdown("## 🍿 Recommendations")
            
            recs_list = st.session_state.recommendations
            if len(recs_list) == 0:
                st.warning("No movies matched your current filters. Try relaxing your filters.")
            else:
                if st.session_state.recommendations_type == 'surprise':
                    st.markdown("### 🎲 Mood Roulette Winner!")
                    
                rows = [recs_list[i:i+4] for i in range(0, len(recs_list), 4)]
                for r_idx, row_movies in enumerate(rows):
                    cols = st.columns(len(row_movies))
                    for idx, movie in enumerate(row_movies):
                        details = tmdb_client.get_movie_details(
                            movie["tmdbId"], 
                            movie["title"], 
                            movie["poster_path"]
                        )
                        
                        with cols[idx]:
                            # Render movie card (HTML clean string output)
                            render_movie_card(movie, details)
                            
                            # Real-time action triggers (➕, 👍, 👎)
                            col_act1, col_act2, col_act3 = st.columns(3)
                            
                            if st.session_state.user_id:
                                # Watchlist Toggle
                                in_wl = db.is_in_watchlist(st.session_state.user_id, int(movie["movieId"]))
                                wl_lbl = "➖" if in_wl else "➕"
                                wl_help = "Remove from Watchlist" if in_wl else "Add to Watchlist"
                                
                                wl_clicked = col_act1.button(wl_lbl, key=f"wl_{movie['movieId']}_{r_idx}_{idx}", help=wl_help)
                                
                                if wl_clicked:
                                    if in_wl:
                                        db.remove_from_watchlist(st.session_state.user_id, int(movie["movieId"]))
                                        st.toast(f"Removed '{movie['clean_title']}' from Watchlist", icon="🗑️")
                                    else:
                                        db.add_to_watchlist(st.session_state.user_id, int(movie["movieId"]), movie["clean_title"], movie["tmdbId"])
                                        st.toast(f"Added '{movie['clean_title']}' to Watchlist!", icon="💖")
                                    time.sleep(0.35)
                                    st.rerun()
                                    
                                # Likes Toggle
                                likes_dict = db.get_user_likes_dislikes(st.session_state.user_id)
                                current_like = likes_dict.get(int(movie["movieId"]), None)
                                like_lbl = "❤️" if current_like == True else "👍"
                                like_help = "Unlike movie" if current_like == True else "Like movie"
                                
                                like_clicked = col_act2.button(like_lbl, key=f"like_{movie['movieId']}_{r_idx}_{idx}", help=like_help)
                                
                                if like_clicked:
                                    new_like = None if current_like == True else True
                                    db.set_like_dislike(st.session_state.user_id, int(movie["movieId"]), new_like)
                                    if new_like:
                                        st.toast(f"Liked '{movie['clean_title']}'!", icon="👍")
                                    else:
                                        st.toast(f"Removed Like from '{movie['clean_title']}'", icon="🗑️")
                                    time.sleep(0.35)
                                    st.rerun()
                                    
                                # Dislike Toggle
                                dislike_lbl = "💔" if current_like == False else "👎"
                                dis_help = "Remove dislike" if current_like == False else "Dislike movie"
                                
                                dis_clicked = col_act3.button(dislike_lbl, key=f"dis_{movie['movieId']}_{r_idx}_{idx}", help=dis_help)
                                
                                if dis_clicked:
                                    new_like = None if current_like == False else False
                                    db.set_like_dislike(st.session_state.user_id, int(movie["movieId"]), new_like)
                                    if new_like is False:
                                        st.toast(f"Disliked '{movie['clean_title']}'", icon="👎")
                                    else:
                                        st.toast(f"Removed Dislike from '{movie['clean_title']}'", icon="🗑️")
                                    time.sleep(0.35)
                                    st.rerun()
                            else:
                                # Sign in prompts for guest explorations
                                col_act1.markdown("<p style='text-align:center; font-size:0.65rem; color:#94a3b8; margin:0;'>Sign in</p>", unsafe_allow_html=True)
                                col_act2.markdown("<p style='text-align:center; font-size:0.65rem; color:#ffd700; margin:0;'>to rate</p>", unsafe_allow_html=True)
                                col_act3.markdown("<p style='text-align:center; font-size:0.65rem; color:#94a3b8; margin:0;'>movies</p>", unsafe_allow_html=True)
                                
                            # Details Expander
                            with st.expander("Details & Trailer"):
                                if st.session_state.user_id:
                                    db.add_watch_history(st.session_state.user_id, int(movie["movieId"]), movie["clean_title"], movie["tmdbId"])
                                    
                                st.write(f"**Director:** {movie['director'] if movie['director'] else 'Unknown'}")
                                st.write(f"**Cast:** {movie['cast_names'] if movie['cast_names'] else 'N/A'}")
                                st.write(movie["overview"] if movie["overview"] else "No overview description available.")
                                
                                if st.session_state.user_id:
                                    user_ratings = db.get_user_ratings(st.session_state.user_id)
                                    cur_rating = user_ratings.get(int(movie["movieId"]), 3.0)
                                    rating_val = st.slider("Rate movie:", 0.5, 5.0, float(cur_rating), 0.5, key=f"rate_sld_{movie['movieId']}_{r_idx}_{idx}")
                                    if st.button("Submit Rating", key=f"rate_btn_{movie['movieId']}_{r_idx}_{idx}"):
                                        db.rate_movie(st.session_state.user_id, int(movie["movieId"]), rating_val)
                                        st.toast(f"Rated '{movie['clean_title']}' {rating_val} stars!", icon="⭐")
                                        time.sleep(0.35)
                                        st.rerun()
                                        
                                st.markdown("---")
                                st.markdown("<p style='font-size:0.82rem; font-weight:bold; margin-bottom:4px; color:#ffd700;'>📊 Score breakdown</p>", unsafe_allow_html=True)
                                st.caption(f"🎭 Mood Match: {movie['mood_match_pct']:.1f}%")
                                st.progress(max(0.0, min(1.0, float(movie['mood_match_pct']) / 100.0)))
                                
                                st.caption(f"👥 Collaborative Fit: {movie['collab_score_pct']:.1f}%")
                                st.progress(max(0.0, min(1.0, float(movie['collab_score_pct']) / 100.0)))
                                
                                if float(movie['personal_score_pct']) > 0:
                                    st.caption(f"💖 Personal Preferences: {movie['personal_score_pct']:.1f}%")
                                    st.progress(max(0.0, min(1.0, float(movie['personal_score_pct']) / 100.0)))
                                    
                                if float(movie['content_match_pct']) > 0:
                                    st.caption(f"🔍 Story Matching: {movie['content_match_pct']:.1f}%")
                                    st.progress(max(0.0, min(1.0, float(movie['content_match_pct']) / 100.0)))
                                    
                                t_url = details["trailer_url"]
                                if "embed" in t_url:
                                    st.markdown(f'<iframe width="100%" height="200" src="{t_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f"[🎥 Search Trailer on YouTube]({t_url})")


    # ----------------- TAB 2: My Watchlist & Profile (Personalized) -----------------
    with tab2:
        if st.session_state.user_id is None:
            st.info("🔓 Please log in from the gateway or sidebar to access your personalized Watchlist, History, and Likes profile!")
        else:
            st.markdown(f"## 💖 Hello, {st.session_state.username}")
            
            col_w1, col_w2 = st.columns([1, 1])
            
            with col_w1:
                with st.container(border=True):
                    st.markdown("### 📋 My Watchlist")
                    watchlist_data = db.get_watchlist(st.session_state.user_id)
                    
                    if not watchlist_data:
                        st.write("Your watchlist is currently empty. Click '➕' on recommendations to fill it!")
                    else:
                        for w in watchlist_data:
                            col_wl_item1, col_wl_item2 = st.columns([4, 1])
                            col_wl_item1.write(f"🎬 **{w['title']}**")
                            if col_wl_item2.button("Remove 🗑️", key=f"rm_wl_{w['movie_id']}"):
                                db.remove_from_watchlist(st.session_state.user_id, int(w["movie_id"]))
                                st.toast("Removed from Watchlist", icon="🗑️")
                                time.sleep(0.3)
                                st.rerun()
                
            with col_w2:
                with st.container(border=True):
                    st.markdown("### 👍 Liked Movies")
                    likes_dict = db.get_user_likes_dislikes(st.session_state.user_id)
                    liked_ids = [m_id for m_id, is_like in likes_dict.items() if is_like]
                    
                    if not liked_ids:
                        st.write("No liked movies logged yet. Click '👍' on recommendation cards!")
                    else:
                        for l_id in liked_ids:
                            match = recommender_model.df[recommender_model.df["movieId"] == l_id]
                            if len(match) > 0:
                                col_l_item1, col_l_item2 = st.columns([4, 1])
                                col_l_item1.write(f"❤️ **{match.iloc[0]['clean_title']}**")
                                if col_l_item2.button("Unlike 🗑️", key=f"rm_l_{l_id}"):
                                    db.set_like_dislike(st.session_state.user_id, int(l_id), None)
                                    st.toast("Removed Like", icon="🗑️")
                                    time.sleep(0.3)
                                    st.rerun()
                
            # Recent History logs
            with st.container(border=True):
                st.markdown("### ⏰ Recent Activity History (Last 20 Watched/Interacted)")
                history_data = db.get_watch_history(st.session_state.user_id, limit=20)
                
                if not history_data:
                    st.write("No activity history tracked yet. Expand movie details or click trailers to log watch events!")
                else:
                    history_table = pd.DataFrame(history_data)
                    history_table.columns = ["Movie ID", "Title", "TMDB ID", "Activity Date/Time"]
                    st.dataframe(history_table, use_container_width=True)


    # ----------------- TAB 3: Analytics Dashboard -----------------
    with tab3:
        st.markdown("## 📊 Recommender Insights & Data Analytics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total Library Movies", f"{len(recommender_model.df):,}")
        col_stat2.metric("MovieLens Ratings", "32,000,204")
        col_stat3.metric("IMDb Reviews Dataset", "50,000")
        col_stat4.metric("SVD Model Latent Factors", "50")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            with st.container(border=True):
                st.markdown("### 🏷️ Most Popular Movie Genres")
                all_genres = recommender_model.df["genres"].str.split("|").explode()
                genre_counts = all_genres.value_counts().reset_index()
                genre_counts.columns = ["Genre", "Count"]
                
                fig = px.bar(
                    genre_counts.head(10), 
                    x="Count", 
                    y="Genre", 
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Purples",
                    height=300
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with col_g2:
            with st.container(border=True):
                st.markdown("### 🏆 Top 10 Most Rated Movies in MovieLens")
                top_rated = recommender_model.df.sort_values(by="rating_count", ascending=False).head(10)
                
                fig = px.bar(
                    top_rated,
                    x="rating_count",
                    y="clean_title",
                    orientation="h",
                    color="avg_rating",
                    color_continuous_scale="RdPu",
                    height=300,
                    labels={"rating_count": "Total Ratings", "clean_title": "Movie Title", "avg_rating": "Avg Rating"}
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with st.container(border=True):
            st.markdown("### 📈 Session Activity Logging")
            
            if st.session_state.user_id:
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("#### Dominant Session Queries Logged")
                    search_data = db.get_search_history(st.session_state.user_id)
                    if not search_data:
                        st.write("No session searches logged yet.")
                    else:
                        st.dataframe(pd.DataFrame(search_data), use_container_width=True)
                with col_h2:
                    st.markdown("#### User Ratings Logged")
                    ratings_data = db.get_user_ratings(st.session_state.user_id)
                    if not ratings_data:
                        st.write("No ratings submitted yet.")
                    else:
                        ratings_rows = []
                        for m_id, r in ratings_data.items():
                            m_title = recommender_model.df[recommender_model.df["movieId"] == m_id].iloc[0]["clean_title"]
                            ratings_rows.append({"Movie": m_title, "Stars": r})
                        st.dataframe(pd.DataFrame(ratings_rows), use_container_width=True)
            else:
                st.info("Log in to track database session logs.")


    # ----------------- TAB 4: About CineMood -----------------
    with tab4:
        with st.container(border=True):
            st.markdown("""
            ## 🎬 CineMood AI System Architecture
            
            CineMood AI is designed to address the missing emotional connection in modern movie recommendation engines. Here is the technical workflow:
            
            1. **Lightweight Emotion Analysis**:
               - Scans user prompts instantly using keyword-matching anchors across **25 detailed emotions**.
               - Maps these emotions to 8 base movie moods (*Happy, Sad, Excited, Relaxed, Inspirational, Romantic, Thoughtful, Suspenseful*) using a static weight matrix.
               - Employs a crisis interceptor safety filter to block recommendations and redirect to suicide lifelines (988) on self-harm inputs.
               
            2. **Fast Sentiment Correction**:
               - Employs a **TF-IDF + Logistic Regression** text classification model trained on **50,000 IMDb movie reviews** achieving **89.76% accuracy**.
               - Gauges positive/negative polarity of the query.
               
            3. **Prompt Story Matching (TF-IDF Cosine Similarity)**:
               - Computes cosine similarity of prompt vocabulary against movie metadata soups (titles, keywords, summaries, genres). Requires zero GPU memory overhead, running in under 10ms!
               
            4. **Collaborative Filtering**:
               - Powered by an **SVD Matrix Factorization model** trained on the **32 Million MovieLens ratings** dataset.
               - Predicts user rating scores based on historical user rating matrices.
               
            5. **Personalized Preference Engine**:
               - Uses a local SQLite database (`data/cinemood.db`) to record watchlist and ratings.
               - Learns user preference vectors dynamically with **weekly time decay**, giving higher weights to recent interactions.
            """)
