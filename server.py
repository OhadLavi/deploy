from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import random
import datetime
import gensim
import numpy as np
import re
import os
import logging
from logging.handlers import RotatingFileHandler
from config import config

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Set up logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/semental.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Semental startup')

# Initialize CORS with all origins allowed
CORS(app, 
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    methods=["GET", "POST", "OPTIONS"]
)

# Initialize Talisman for security headers
Talisman(app, 
    force_https=app.config.get('SSL_REDIRECT', False),
    content_security_policy=app.config['SECURE_HEADERS']['Content-Security-Policy']
)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config['RATELIMIT_DEFAULT']]
)

def is_hebrew_word(word):
    """Check if a word contains only Hebrew characters."""
    return bool(re.match(r'^[\u0590-\u05FF]+$', word))

# Load words from words.txt
def load_words():
    try:
        with open(app.config['WORDS_PATH'], 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip() and is_hebrew_word(line.strip())]
    except Exception as e:
        app.logger.error(f"Error loading words from file: {e}")
        return []

# Load the Word2Vec model and filter Hebrew words
try:
    model = gensim.models.KeyedVectors.load_word2vec_format(
        app.config['MODEL_PATH'] + '.bin',
        binary=True
    )
    hebrew_words = load_words()
    hebrew_vocab = [word for word in hebrew_words if word in model]
    if not hebrew_vocab:
        raise ValueError("No Hebrew words found in both file and model vocabulary")
except Exception as e:
    app.logger.error(f"Error loading model or filtering Hebrew words: {e}")
    exit(1)

@app.route("/health")
@limiter.exempt
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat()
    })

# Add a root route that returns available endpoints
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        'message': 'Semental Server is running',
        'status': 'OK',
        'available_endpoints': [
            '/api/game/daily/stats',
            '/api/game/casual/stats',
            '/api/game/daily/guess',
            '/api/game/casual/guess',
            '/api/game/casual/reset',
            '/api/game/daily/reveal',
            '/api/game/casual/reveal',
            '/api/game/daily/hint',
            '/api/game/casual/hint'
        ]
    })

# Get the reference date for daily game numbering
GAME_START_DATE = datetime.date(2024, 2, 21)  # When the game was launched

def get_daily_secret():
    """Get the daily secret word based on the date."""
    today = datetime.date.today()
    days_since_start = (today - GAME_START_DATE).days
    return hebrew_vocab[days_since_start % len(hebrew_vocab)]

daily_secret = get_daily_secret()
casual_secret = random.choice(hebrew_vocab)

def get_similarity_and_rank(secret, guess):
    """
    Computes the cosine similarity (scaled to 0–100) between the secret word and the guess.
    Also returns the rank (position in the top-1000 similar words) if the guess is found.
    The rank is calculated as follows:
    - The target word (answer) gets rank 1000/1000
    - The most similar word (first in array) gets rank 999/1000
    - The 10th most similar word gets rank 990/1000
    - The least similar word (last in array) gets rank 1/1000
    """
    if guess == secret:
        return 100.0, 1000  # The target word always gets 100% similarity and rank 1000
    
    if not is_hebrew_word(guess) or guess not in model:
        return None, None
        
    # Cosine similarity (typically between -1 and 1) scaled to 0-100.
    similarity = model.similarity(secret, guess) * 100
    
    # Get similar words but filter to only Hebrew words and sort by similarity
    similar_words = [(word, sim) for word, sim in model.most_similar(secret, topn=2000) 
                    if is_hebrew_word(word)][:1000]
    
    # Find the word's position in the sorted list
    for i, (word, sim) in enumerate(similar_words):
        if word == guess:
            # Special cases for notable ranks
            if i == 0:  # Most similar word
                rank = 999
            elif i == 9:  # 10th most similar word
                rank = 990
            elif i == len(similar_words) - 1:  # Least similar word
                rank = 1
            else:
                rank = 999 - i  # Normal ranking
            break
    else:
        rank = None
    
    return similarity, rank

def get_game_stats(secret):
    """
    Computes game stats by retrieving the top 1000 most similar words to the secret.
    Returns:
    - The target word (rank 1000)
    - The most similar word (rank 999)
    - The 10th most similar word (rank 990)
    - The least similar word (rank 1)
    """
    # Get similar words but filter to only Hebrew words and sort by similarity
    similar_words = [(word, sim) for word, sim in model.most_similar(secret, topn=2000) 
                    if is_hebrew_word(word)][:1000]
    
    # Calculate game number as days since reference date
    today = datetime.date.today()
    game_number = (today - GAME_START_DATE).days + 1  # +1 so first game is #1 not #0
    
    return {
        "game_number": game_number,
        "closest_word": similar_words[0][0],  # Most similar word (rank 999)
        "closest_similarity": similar_words[0][1] * 100,
        "closest_rank": 999,  # Most similar word gets rank 999
        "tenth_word": similar_words[9][0] if len(similar_words) >= 10 else None,  # 10th most similar (rank 990)
        "tenth_similarity": similar_words[9][1] * 100 if len(similar_words) >= 10 else None,
        "tenth_rank": 990 if len(similar_words) >= 10 else None,
        "thousandth_word": similar_words[-1][0] if len(similar_words) >= 1000 else None,  # Least similar (rank 1)
        "thousandth_similarity": similar_words[-1][1] * 100 if len(similar_words) >= 1000 else None,
        "thousandth_rank": 1 if len(similar_words) >= 1000 else None,
    }

# Endpoint: Get Game Statistics
@app.route("/api/game/<mode>/stats", methods=["GET"])
def game_stats(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    secret = daily_secret if mode == "daily" else casual_secret
    stats = get_game_stats(secret)
    return jsonify(stats)

# Endpoint: Make a Guess
@app.route("/api/game/<mode>/guess", methods=["POST"])
def guess_word(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    data = request.get_json()
    if not data or "guess" not in data:
        abort(400, "Missing guess parameter")
    guess = data["guess"].strip()
    
    # Only allow single-word guesses (no spaces)
    if " " in guess:
        return jsonify({"valid": False, "message": "Guess must be a single word"}), 400

    # Validate that the guess is a Hebrew word
    if not is_hebrew_word(guess):
        return jsonify({"valid": False, "message": "Please enter Hebrew words only"}), 400

    secret = daily_secret if mode == "daily" else casual_secret

    if guess not in model:
        return jsonify({"valid": False, "message": "Word not found in dictionary"}), 400

    similarity, rank = get_similarity_and_rank(secret, guess)
    is_correct = (guess == secret)
    response = {
        "similarity": similarity,
        "rank": rank,
        "is_correct": is_correct,
        "valid": True
    }
    return jsonify(response)

# Endpoint: Reset Casual Game
@app.route("/api/game/casual/reset", methods=["POST"])
def reset_game():
    global casual_secret
    casual_secret = random.choice(hebrew_vocab)
    print(casual_secret)
    return jsonify({"message": "Casual game reset"}), 200

# Endpoint: Reveal the Secret Word
@app.route("/api/game/<mode>/reveal", methods=["GET"])
def reveal_word(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    secret = daily_secret if mode == "daily" else casual_secret
    return jsonify({"word": secret})

# Endpoint: Get a Hint
@app.route("/api/game/<mode>/hint", methods=["GET"])
def get_hint(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    
    secret = daily_secret if mode == "daily" else casual_secret
    
    # Choose a random position in the word
    position = random.randint(0, len(secret) - 1)
    letter = secret[position]
    
    # Create a hint message showing the letter's position
    hint = f"האות במיקום {position + 1} היא '{letter}'"
    
    return jsonify({"hint": hint})

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded", message=str(e.description)), 429

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify(error="Internal server error"), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify(error="Not found"), 404

if __name__ == "__main__":
    # In production, use gunicorn instead
    if app.config['DEBUG']:
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        app.run(host="0.0.0.0", port=5000, ssl_context='adhoc')
