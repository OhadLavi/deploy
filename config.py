import os
from datetime import timedelta

class BaseConfig:
    # Base configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    WORDS_PATH = os.environ.get('WORDS_PATH', os.path.join(os.path.dirname(__file__), 'words.txt'))
    
    # Rate limiting
    RATELIMIT_DEFAULT = "100 per minute"
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    
    # Security headers
    SECURE_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'",
    }

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8080']
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'model'))

class ProductionConfig(BaseConfig):
    DEBUG = False
    SSL_REDIRECT = True
    MODEL_PATH = os.environ.get('MODEL_PATH', '/opt/render/project/src/model/model')
    CORS_ORIGINS = [
        os.environ.get('ALLOWED_ORIGIN', 'https://your-domain.com'),
        'http://localhost:8080',  # Temporarily allow localhost for testing
    ]
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

class TestingConfig(BaseConfig):
    TESTING = True
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8080']
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'model'))

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 