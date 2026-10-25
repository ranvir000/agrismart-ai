import os
from flask import Flask, render_template
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
from werkzeug.security import generate_password_hash
from datetime import datetime

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
    }

    with db.engine.begin() as connection:
        for name, column_type in additions.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE scan ADD COLUMN {name} {column_type}'))


def create_app(test_config=None):
    app = Flask(__name__)
    
    if test_config is None:
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-agrismart-secret-key')

        # A persistent Postgres database is used when DATABASE_URL is set (Render),
        # otherwise we fall back to a local SQLite file.
        database_url = os.environ.get('DATABASE_URL', '').strip()
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///agrismart.db'
        if database_url:
            app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
                'pool_pre_ping': True,
                'pool_recycle': 280,
            }
    else:
        app.config.update(test_config)
        
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Max upload size: 10MB
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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
        return "File too large. Max size is 10MB.", 413

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
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    if not User.query.filter_by(email=admin_email).first():
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
