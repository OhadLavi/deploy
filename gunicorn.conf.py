import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = int(os.environ.get('WEB_CONCURRENCY', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'
worker_connections = 1000
timeout = int(os.environ.get('TIMEOUT', 30))
keepalive = 2

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = os.environ.get('LOG_LEVEL', 'info')

# Process naming
proc_name = 'semental'

# SSL (if not using reverse proxy)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'

# Environment variables
raw_env = [
    f"FLASK_ENV={os.environ.get('FLASK_ENV', 'production')}",
    f"SECRET_KEY={os.environ.get('SECRET_KEY', 'change-in-production')}",
    f"ALLOWED_ORIGIN={os.environ.get('ALLOWED_ORIGIN', 'https://your-domain.com')}",
    f"REDIS_URL={os.environ.get('REDIS_URL', '')}",
] 