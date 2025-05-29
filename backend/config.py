import tempfile
import os

class Config:
    UPLOAD_FOLDER = tempfile.mkdtemp()
    ALLOWED_EXTENSIONS = {'csv'}
    CORS_CONFIG = {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }