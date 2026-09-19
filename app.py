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

def create_app(test_config=None):
    app = Flask(__name__)
    
    if test_config is None:
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-agrismart-secret-key')
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agrismart.db'
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
        seed_admin()

    return app


def seed_admin():
    if not User.query.filter_by(email='admin@agrismart.com').first():
        admin = User(
            username='admin',
            email='admin@agrismart.com',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            city='Delhi',
            farm_name='AgriSmart HQ'
        )
        db.session.add(admin)
        db.session.commit()

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
