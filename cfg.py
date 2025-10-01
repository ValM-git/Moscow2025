import os, secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
UPLOAD_FOLDER_TYPES = os.path.join(UPLOAD_ROOT, "types")
UPLOAD_FOLDER_ITEMS = os.path.join(UPLOAD_ROOT, "items")
UPLOAD_FOLDER_REQ   = os.path.join(UPLOAD_ROOT, "requests")

SQLITE_FILENAME = "tooltrack.db"
SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, SQLITE_FILENAME)

def init_app(app):
    app.config['SECRET_KEY'] = app.config.get('SECRET_KEY') or ('dev-' + secrets.token_hex(8))
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER_TYPES'] = UPLOAD_FOLDER_TYPES
    app.config['UPLOAD_FOLDER_ITEMS'] = UPLOAD_FOLDER_ITEMS
    app.config['UPLOAD_FOLDER_REQ']   = UPLOAD_FOLDER_REQ
    for p in (UPLOAD_FOLDER_TYPES, UPLOAD_FOLDER_ITEMS, UPLOAD_FOLDER_REQ):
        os.makedirs(p, exist_ok=True)


# --- веса (общие) ---
DEFAULT_WEIGHTS = dict(shape=0.75, feat=0.10, color=0.10, ssim=0.05)

# --- веса для «Наборов» (SSIM-центрично) ---
KIT_WEIGHTS     = dict(shape=0.05, feat=0.05, color=0.15, ssim=0.75)
