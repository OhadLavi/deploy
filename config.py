import os
from datetime import timedelta

class BaseConfig:
    # Base configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    MODEL_PATH = os.path.join('/opt/render/project/src/model', 'model.mdl')
    WORDS_PATH = os.path.join(os.path.dirname(__file__), 'words.txt')
    
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
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']

class ProductionConfig(BaseConfig):
    DEBUG = False
    CORS_ORIGINS = [os.environ.get('ALLOWED_ORIGIN', 'https://your-domain.com')]
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # SSL/HTTPS
    SSL_REDIRECT = True

class TestingConfig(BaseConfig):
    TESTING = True
    CORS_ORIGINS = ['http://localhost:3000']

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 