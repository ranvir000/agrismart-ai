import logging
import os
from flask import Flask, render_template, abort, request as flask_request
from flask_login import LoginManager
from database.db import db, User
from routes.auth import auth_bp
from routes.main import main_bp
from routes.dashboard import dashboard_bp
from routes.scan import scan_bp
from routes.history import history_bp
from routes.analytics import analytics_bp
from routes.profile import profile_bp
from routes.diseases import diseases_bp
from routes.admin import admin_bp
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta

def normalize_database_url(database_url):
    """Use psycopg v3 for Render/PostgreSQL URLs, matching requirements.txt."""
    if database_url.startswith('postgres://'):
        return database_url.replace('postgres://', 'postgresql+psycopg://', 1)
    if database_url.startswith('postgresql://'):
        return database_url.replace('postgresql://', 'postgresql+psycopg://', 1)
    return database_url


def ensure_schema():
    """Add columns introduced after a database was first created.

    db.create_all() only creates missing tables, so an existing scan table would
    otherwise reject the newer fields used by the scan route.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if 'scan' not in inspector.get_table_names():
        return

    existing = {column['name'] for column in inspector.get_columns('scan')}
    blob_type = 'BLOB' if db.engine.dialect.name == 'sqlite' else 'BYTEA'
    additions = {
        'image_data': blob_type,
        'image_mimetype': 'VARCHAR(50)',
        'alternatives': 'TEXT',
        'analysis_source': 'VARCHAR(20)',
        'engine_outputs': 'TEXT',
    }

    with db.engine.begin() as connection:
        for name, column_type in additions.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE scan ADD COLUMN {name} {column_type}'))


def create_app(test_config=None):
    app = Flask(__name__)
    
    if test_config is None:
        secret_key = os.environ.get('SECRET_KEY')
        is_production = (os.environ.get('FLASK_ENV', '').lower() == 'production'
                         or bool(os.environ.get('RENDER')))
        if is_production and not secret_key:
            raise RuntimeError('SECRET_KEY must be configured in production.')
        app.config['SECRET_KEY'] = secret_key or os.urandom(32)

        # A persistent Postgres database is used when DATABASE_URL is set (Render),
        # otherwise we fall back to a local SQLite file.
        database_url = os.environ.get('DATABASE_URL', '').strip()
        database_url = normalize_database_url(database_url)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///agrismart.db'
        if database_url:
            app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
                'pool_pre_ping': True,
                'pool_recycle': 280,
            }
    else:
        app.config.update(test_config)
        
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # A signed-in browser must not stay signed in indefinitely. Flask-Login's
    # "remember me" cookie defaults to 365 days, which made the app open straight
    # into a previously used account even after weeks. Both are capped here.
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=12)
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=1)
    app.config['REMEMBER_COOKIE_HTTPONLY'] = True
    app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
    app.config.setdefault(
        'REMEMBER_COOKIE_SECURE',
        (os.environ.get('FLASK_ENV', '').lower() == 'production' or bool(os.environ.get('RENDER')))
    )
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config.setdefault(
        'SESSION_COOKIE_SECURE',
        (os.environ.get('FLASK_ENV', '').lower() == 'production' or bool(os.environ.get('RENDER')))
    )
    # Max upload size: 16MB for large phone photos and TIFF/BMP scans. Uploads are
    # re-encoded to a compact JPEG before storage, so this does not bloat the DB.
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    # Cross-site form posts are rejected by a same-origin check. Browsers always send
    # an Origin (or at least a Referer) header on POST, so a write coming from another
    # site is refused before it reaches a view. Requests carrying neither header (the
    # test client, command-line tools) are unaffected, and every app form is same-origin.
    @app.before_request
    def reject_cross_site_writes():
        if flask_request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
            return None
        origin = flask_request.headers.get('Origin') or flask_request.headers.get('Referer') or ''
        if not origin:
            return None
        origin_host = origin.split('://')[-1].split('/')[0].strip()
        if origin_host and origin_host != flask_request.host:
            abort(403)
        return None

    @app.before_request
    def block_public_legacy_uploads():
        # Old versions wrote user photos under static/uploads, where Flask's static
        # route would expose them without authentication. Keep files for legacy DB
        # records, but serve them only through the authenticated scan-image endpoint.
        if flask_request.path.startswith('/static/uploads/'):
            abort(404)

    db.init_app(app)
    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(scan_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(diseases_bp)
    app.register_blueprint(admin_bp)

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('403.html'), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return "File too large. Max size is 16MB.", 413

    @app.context_processor
    def inject_now():
        return {'now': datetime.utcnow()}

    with app.app_context():
        db.create_all()
        ensure_schema()
        seed_admin()

    return app


def seed_admin():
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@agrismart.com')
    admin_password = os.environ.get('ADMIN_PASSWORD')
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    existing_admin = User.query.filter_by(email=admin_email).first()
    if admin_password == 'admin123':
        raise RuntimeError('ADMIN_PASSWORD must not use the public legacy default.')
    if not admin_password:
        if existing_admin and check_password_hash(existing_admin.password_hash, 'admin123'):
            raise RuntimeError(
                'A legacy administrator still uses the public default password. '
                'Configure ADMIN_PASSWORD to rotate it before startup.'
            )
        logging.getLogger(__name__).warning(
            'ADMIN_PASSWORD is not configured; skipping administrator bootstrap.'
        )
        return
    if existing_admin:
        if check_password_hash(existing_admin.password_hash, 'admin123'):
            existing_admin.password_hash = generate_password_hash(admin_password)
            db.session.commit()
            logging.getLogger(__name__).warning(
                'Rotated a legacy administrator password from the public default.'
            )
        return
    if not existing_admin:
        admin = User(
            username=admin_username,
            email=admin_email,
            password_hash=generate_password_hash(admin_password),
            role='admin',
            city='Delhi',
            farm_name='AgriSmart HQ'
        )
        db.session.add(admin)
        db.session.commit()

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
