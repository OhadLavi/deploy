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
    resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 120  # Cache preflight response for 2 minutes
    }},
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
    # Try to load the binary model first
    model_path = app.config['MODEL_PATH']
    app.logger.info(f"Attempting to load model from {model_path}")
    
    try:
        # Try loading as binary format
        model = gensim.models.KeyedVectors.load_word2vec_format(
            f"{model_path}.bin",
            binary=True
        )
        app.logger.info("Loaded model in binary format")
    except Exception as e:
        app.logger.warning(f"Failed to load binary model: {e}")
        try:
            # Try loading as regular model
            model = gensim.models.KeyedVectors.load(model_path)
            app.logger.info("Loaded model in regular format")
        except Exception as e2:
            app.logger.warning(f"Failed to load regular model: {e2}")
            # Try loading as word2vec text format
            model = gensim.models.KeyedVectors.load_word2vec_format(
                f"{model_path}.txt",
                binary=False
            )
            app.logger.info("Loaded model in text format")
    
    hebrew_words = load_words()
    app.logger.info(f"Loaded {len(hebrew_words)} Hebrew words from file")
    
    hebrew_vocab = [word for word in hebrew_words if word in model]
    app.logger.info(f"Found {len(hebrew_vocab)} Hebrew words in model vocabulary")
    
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

# Store casual secrets by session ID with timestamps
casual_secrets = {}

def cleanup_old_sessions():
    """Clean up sessions older than 24 hours."""
    now = datetime.datetime.now()
    expired = [
        sid for sid, (_, timestamp) in casual_secrets.items()
        if (now - timestamp).days >= 1
    ]
    for sid in expired:
        del casual_secrets[sid]

def get_casual_secret(session_id):
    """Get or create a casual secret word for the given session ID."""
    cleanup_old_sessions()  # Cleanup old sessions
    
    if session_id not in casual_secrets:
        casual_secrets[session_id] = (random.choice(hebrew_vocab), datetime.datetime.now())
        app.logger.info(f"Created new casual secret for session {session_id}: {casual_secrets[session_id][0]}")
    else:
        # Update timestamp
        word, _ = casual_secrets[session_id]
        casual_secrets[session_id] = (word, datetime.datetime.now())
    
    return casual_secrets[session_id][0]

def clear_casual_secret(session_id):
    """Clear a casual secret for the given session ID."""
    if session_id in casual_secrets:
        del casual_secrets[session_id]

# Add a global dictionary to store pre-calculated top words for each secret
top_words_cache = {}

def get_top_similar_words(secret, count=1000):
    """Get the top similar words for a secret word."""
    if secret in top_words_cache:
        return top_words_cache[secret]
    
    # Get similar words but filter to only Hebrew words and sort by similarity
    similar_words = [(word, sim * 100) for word, sim in model.most_similar(secret, topn=count*2) 
                    if is_hebrew_word(word)][:count]
    
    # Store in cache
    top_words_cache[secret] = similar_words
    return similar_words

def get_similarity_and_rank(secret, guess):
    """
    Computes the cosine similarity (scaled to 0–100) between the secret word and the guess.
    Also returns the rank based on the pre-calculated top words list.
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
    
    # Get direct similarity
    similarity = model.similarity(secret, guess) * 100
    
    # Get the pre-calculated top words
    top_words = get_top_similar_words(secret)
    
    # Check if the guess is in the top words
    for i, (word, sim) in enumerate(top_words):
        if word == guess:
            # Special cases for notable ranks
            if i == 0:  # Most similar word
                rank = 999
            elif i == 9:  # 10th most similar word
                rank = 990
            elif i == len(top_words) - 1:  # Least similar word
                rank = 1
            else:
                rank = 999 - i  # Normal ranking
            return similarity, rank
    
    # If not in top words, check if it would rank based on similarity
    # Find where it would fit in the sorted list
    position = len(top_words)  # Default to after the last word
    for i, (word, sim) in enumerate(top_words):
        if similarity > sim:
            position = i
            break
    
    # If it would be in the top 1000, calculate its rank
    if position < 1000:
        # Special cases for notable positions
        if position == 0:
            rank = 999
        elif position == 9:
            rank = 990
        else:
            rank = 999 - position
        return similarity, rank
    else:
        # Not in top 1000
        return similarity, None

def get_game_stats(secret):
    """
    Computes game stats by retrieving the top 1000 most similar words to the secret.
    Returns:
    - The target word (rank 1000)
    - The most similar word (rank 999)
    - The 10th most similar word (rank 990)
    - The least similar word (rank 1)
    """
    # Get the pre-calculated top words
    top_words = get_top_similar_words(secret)
    
    # Calculate game number as days since reference date
    today = datetime.date.today()
    game_number = (today - GAME_START_DATE).days + 1  # +1 so first game is #1 not #0
    
    return {
        "game_number": game_number,
        "closest_word": top_words[0][0],  # Most similar word (rank 999)
        "closest_similarity": top_words[0][1],
        "closest_rank": 999,  # Most similar word gets rank 999
        "tenth_word": top_words[9][0] if len(top_words) >= 10 else None,  # 10th most similar (rank 990)
        "tenth_similarity": top_words[9][1] if len(top_words) >= 10 else None,
        "tenth_rank": 990 if len(top_words) >= 10 else None,
        "thousandth_word": top_words[-1][0] if len(top_words) >= 1000 else None,  # Least similar (rank 1)
        "thousandth_similarity": top_words[-1][1] if len(top_words) >= 1000 else None,
        "thousandth_rank": 1 if len(top_words) >= 1000 else None,
    }

# Endpoint: Get Game Statistics
@app.route("/api/game/<mode>/stats", methods=["GET"])
def game_stats(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    
    # For casual mode, require a session ID
    if mode == "casual":
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "Missing session_id parameter"}), 400
        secret = get_casual_secret(session_id)
    else:
        secret = daily_secret
    
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
    
    # For casual mode, require a session ID
    if mode == "casual":
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "Missing session_id parameter"}), 400
        secret = get_casual_secret(session_id)
    else:
        secret = daily_secret
    
    # Only allow single-word guesses (no spaces)
    if " " in guess:
        return jsonify({"valid": False, "message": "Guess must be a single word"}), 400

    # Validate that the guess is a Hebrew word
    if not is_hebrew_word(guess):
        return jsonify({"valid": False, "message": "Please enter Hebrew words only"}), 400

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
    data = request.get_json()
    if not data or "session_id" not in data:
        return jsonify({"error": "Missing session_id parameter"}), 400
    
    session_id = data["session_id"]
    
    # Clear the cache for the old secret if it exists
    if session_id in casual_secrets:
        clear_word_cache(casual_secrets[session_id][0])
        clear_casual_secret(session_id)
    
    # Generate new secret
    secret = get_casual_secret(session_id)
    
    return jsonify({
        "message": "Casual game reset",
        "session_id": session_id
    }), 200

# Endpoint: Reveal the Secret Word
@app.route("/api/game/<mode>/reveal", methods=["GET"])
def reveal_word(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    
    # For casual mode, require a session ID
    if mode == "casual":
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "Missing session_id parameter"}), 400
        secret = get_casual_secret(session_id)
    else:
        secret = daily_secret
    
    return jsonify({"word": secret})

# Endpoint: Get a Hint
@app.route("/api/game/<mode>/hint", methods=["GET"])
def get_hint(mode):
    if mode not in ["daily", "casual"]:
        abort(404)
    
    # For casual mode, require a session ID
    if mode == "casual":
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "Missing session_id parameter"}), 400
        secret = get_casual_secret(session_id)
    else:
        secret = daily_secret
    
    # Choose a random position in the word
    position = random.randint(0, len(secret) - 1)
    letter = secret[position]
    
    # Create a hint message showing the letter's position
    hint = f"האות במיקום {position + 1} היא '{letter}'"
    
    return jsonify({"hint": hint})

# Add a function to clear the cache for memory management
def clear_word_cache(secret=None):
    """Clear the word cache for a specific secret or all secrets."""
    global top_words_cache
    if secret:
        if secret in top_words_cache:
            del top_words_cache[secret]
    else:
        top_words_cache = {}

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
        app.run(host="0.0.0.0", port=8080, debug=True)
    else:
        app.run(host="0.0.0.0", port=8080, ssl_context='adhoc')
