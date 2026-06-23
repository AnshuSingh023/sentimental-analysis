import os
import sys
import numpy as np
import re

# Optimized keyword mappings for 25 detailed emotions
EMOTION_KEYWORDS = {
    "Happy": ["happy", "cheerful", "glad", "merry", "delighted", "good mood", "good day", "joyous", "smiling"],
    "Joyful": ["joyful", "joy", "gleeful", "overjoyed", "jubilant", "ecstatic", "beaming", "rejoicing"],
    "Excited": ["excited", "thrilled", "eager", "enthusiastic", "animated", "pumped", "elated", "hyped"],
    "Romantic": ["romantic", "romance", "love", "passionate", "affectionate", "lovey-dovey", "dating", "in love", "sweetheart"],
    "Relaxed": ["relaxed", "relax", "chill", "resting", "laid-back", "unwind", "easygoing", "comfortable", "cozy"],
    "Calm": ["calm", "peaceful", "serene", "tranquil", "quiet", "still", "composed", "peace of mind"],
    "Inspired": ["inspired", "inspirational", "uplifted", "creative", "stimulated", "encouraged", "moved", "touched"],
    "Motivated": ["motivated", "motivate", "determination", "ambitious", "focused", "driven", "purposeful", "persistent", "energetic"],
    "Curious": ["curious", "interested", "inquisitive", "wondering", "fascinated", "intrigued", "exploring"],
    "Adventurous": ["adventurous", "adventure", "daring", "bold", "venturesome", "exploring", "thrill-seeking", "wild", "brave"],
    "Thoughtful": ["thoughtful", "reflective", "contemplative", "pensive", "meditative", "deep thinking", "intellectual"],
    "Nostalgic": ["nostalgic", "nostalgia", "sentimental", "longing for the past", "remembering old days", "reminiscing", "childhood"],
    "Lonely": ["lonely", "lonesome", "isolated", "companionless", "alone", "friendless", "abandoned"],
    "Sad": ["sad", "unhappy", "sorrowful", "down", "gloomy", "blue", "tearful", "heartache", "miserable"],
    "Heartbroken": ["heartbroken", "heartbreak", "devastated", "crushed", "grieved", "broken-hearted", "breakup"],
    "Grieving": ["grieving", "mourning", "loss of loved one", "bereaved", "grief-stricken", "deep sorrow", "lamenting"],
    "Hopeless": ["hopeless", "despairing", "defeatist", "downhearted", "futile", "giving up", "pessimistic"],
    "Depressed": ["depressed", "depression", "melancholy", "desolate", "empty inside", "heavy-hearted", "downcast"],
    "Anxious": ["anxious", "nervous", "uneasy", "apprehensive", "worried", "tense", "fearful", "jittery"],
    "Stressed": ["stressed", "overwhelmed", "burdened", "pressured", "stressed out", "exhausted", "fatigued"],
    "Angry": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "outraged", "enraged", "pissed off", "resentful"],
    "Fearful": ["fearful", "scared", "afraid", "frightened", "terrified", "spooked", "horror-struck", "dread"],
    "Suspenseful": ["suspenseful", "tense", "mysterious", "thrilling", "on edge", "anticipating", "gripping", "suspense"],
    "Melancholic": ["melancholic", "melancholy", "somber", "wistful", "pensive sadness", "low-spirited"],
    "Emotional": ["emotional", "sensitive", "deeply moved", "touchy", "vulnerable", "tearful", "passionate"]
}

# Mapping weights from 25 emotions to 8 base movie moods
EMOTION_TO_MOOD_WEIGHTS = {
    "Happy": {"Happy": 0.8, "Inspirational": 0.2},
    "Joyful": {"Happy": 0.8, "Inspirational": 0.2},
    "Excited": {"Excited": 0.8, "Happy": 0.2},
    "Romantic": {"Romantic": 0.9, "Happy": 0.1},
    "Relaxed": {"Relaxed": 1.0},
    "Calm": {"Relaxed": 0.7, "Thoughtful": 0.3},
    "Inspired": {"Inspirational": 0.8, "Thoughtful": 0.2},
    "Motivated": {"Inspirational": 0.6, "Excited": 0.4},
    "Curious": {"Thoughtful": 0.7, "Excited": 0.3},
    "Adventurous": {"Excited": 0.7, "Thoughtful": 0.3},
    "Thoughtful": {"Thoughtful": 0.9, "Relaxed": 0.1},
    "Nostalgic": {"Thoughtful": 0.5, "Relaxed": 0.3, "Sad": 0.2},
    "Lonely": {"Sad": 0.8, "Thoughtful": 0.2},
    "Sad": {"Sad": 0.9, "Thoughtful": 0.1},
    "Heartbroken": {"Sad": 0.9, "Thoughtful": 0.1},
    "Grieving": {"Sad": 0.9, "Thoughtful": 0.1},
    "Hopeless": {"Sad": 0.8, "Thoughtful": 0.2},
    "Depressed": {"Sad": 0.8, "Thoughtful": 0.2},
    "Anxious": {"Suspenseful": 0.6, "Sad": 0.4},
    "Stressed": {"Suspenseful": 0.4, "Sad": 0.4, "Relaxed": 0.2},
    "Angry": {"Suspenseful": 0.7, "Sad": 0.3},
    "Fearful": {"Suspenseful": 0.9, "Sad": 0.1},
    "Suspenseful": {"Suspenseful": 0.9, "Thoughtful": 0.1},
    "Melancholic": {"Sad": 0.6, "Thoughtful": 0.4},
    "Emotional": {"Sad": 0.4, "Romantic": 0.3, "Happy": 0.3}
}

# Crisis verification regexes (fast, zero memory overhead)
CRISIS_REGEX_PATTERNS = [
    r"\b(kill|end|terminate|destroy|harm)\s+myself\b",
    r"\b(want|wish|decide|going)\s+to\s+(die|suicide|kill\s+myself)\b",
    r"\b(dont|don't|do\s+not)\s+want\s+to\s+live\b",
    r"\b(suicidal|self\s*harm|cutting\s+myself|overdose\s+myself|hanging\s+myself)\b",
    r"\bi\s+want\s+to\s+die\b",
    r"\bi\s+want\s+to\s+kill\s+myself\b",
    r"\bfeeling\s+like\s+ending\s+my\s+life\b"
]

class CineMoodEmotionDetector:
    def __init__(self, use_transformer=False):
        # Disabled transformers permanently to make the project light on memory
        self.use_transformer = False
        print("CineMoodEmotionDetector initialized in lightweight lexical mode.")

    def check_crisis(self, text):
        """Validates text input for safety issues / self-harm indicators."""
        text_clean = str(text).lower().strip()
        for pattern in CRISIS_REGEX_PATTERNS:
            if re.search(pattern, text_clean):
                return True
        return False

    def get_basic_emotions(self, text):
        """Computes probability distribution across all 25 basic emotions instantly."""
        text_clean = str(text).lower()
        scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS.keys()}
        total = 0.0
        
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw)
                matches = len(re.findall(pattern, text_clean))
                if matches > 0:
                    scores[emotion] += matches
                    total += matches
                    
        if total == 0:
            # Uniform distribution fallback
            return {emotion: 1.0 / len(EMOTION_KEYWORDS) for emotion in EMOTION_KEYWORDS.keys()}
            
        return {emotion: score / total for emotion, score in scores.items()}

    def get_mood_scores(self, text):
        """Processes text to find crisis triggers, 25 emotions, and maps to 8 movie moods."""
        if self.check_crisis(text):
            mood_scores = {m: 0.0 for m in ["Happy", "Sad", "Excited", "Relaxed", "Inspirational", "Romantic", "Thoughtful", "Suspenseful"]}
            mood_scores["Sad"] = 0.6
            mood_scores["Thoughtful"] = 0.4
            
            basic_emotions = {e: 0.0 for e in EMOTION_KEYWORDS.keys()}
            basic_emotions["Hopeless"] = 0.4
            basic_emotions["Depressed"] = 0.3
            basic_emotions["Sad"] = 0.3
            
            return {
                "mood_scores": mood_scores,
                "dominant_mood": "Sad",
                "primary_emotion": "Hopeless",
                "secondary_emotion": "Depressed",
                "confidence": 1.0,
                "basic_emotions": basic_emotions,
                "is_crisis": True
            }
            
        basic_emotions = self.get_basic_emotions(text)
        
        # Sort emotions
        sorted_emotions = sorted(basic_emotions.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0]
        secondary_emotion = sorted_emotions[1][0]
        
        # Map to 8 moods
        mood_scores = {m: 0.0 for m in ["Happy", "Sad", "Excited", "Relaxed", "Inspirational", "Romantic", "Thoughtful", "Suspenseful"]}
        for emotion, prob in basic_emotions.items():
            if emotion in EMOTION_TO_MOOD_WEIGHTS:
                for mood, weight in EMOTION_TO_MOOD_WEIGHTS[emotion].items():
                    mood_scores[mood] += prob * weight
                    
        # Normalize mood scores to sum to 1.0
        total_mood = sum(mood_scores.values())
        if total_mood > 0:
            mood_scores = {m: s / total_mood for m, s in mood_scores.items()}
        else:
            mood_scores = {m: 0.125 for m in mood_scores.keys()}
            
        dominant_mood = max(mood_scores, key=mood_scores.get)
        
        return {
            "mood_scores": mood_scores,
            "dominant_mood": dominant_mood,
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "confidence": float(sorted_emotions[0][1]),
            "basic_emotions": basic_emotions,
            "is_crisis": False
        }

if __name__ == "__main__":
    detector = CineMoodEmotionDetector()
    print(detector.get_mood_scores("I want a happy and funny comedy to lift my spirits."))
