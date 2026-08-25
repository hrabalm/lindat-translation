import logging
import time
import uuid

from flask import Flask, Blueprint, g, request
from . import settings
from .extensions import bootstrap
from .main.views import bp as main
from app.main.api.restplus import api
from app.main.api.translation.endpoints.models import ns as models_ns
from app.main.api.translation.endpoints.languages import ns as languages_ns
from app.main.api.translation.endpoints.root import ns as root_ns


def configure_application_logging():
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            '%(levelname)s:%(name)s:%(message)s'
        ))
        logger.addHandler(handler)
    logger.propagate = False


class ReverseProxied(object):
    '''Wrap the application in this middleware and configure the 
    front-end server to add these headers, to let you quietly bind 
    this to a URL other than / and to an HTTP scheme that is 
    different than what is used locally.

    In nginx:
    location /myprefix {
        proxy_pass http://192.168.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /myprefix;
        }

    :param app: the WSGI application
    '''
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)

def create_app():
    configure_application_logging()
    app = Flask(__name__)
    app.wsgi_app = ReverseProxied(app.wsgi_app)
    app.config.from_object(settings)
    app.config.from_envvar('LOCAL_SETTINGS', silent=True)
    logging.getLogger().error('DEFAULT_SERVER=' + app.config.get('DEFAULT_SERVER'))
    bootstrap.init_app(app)
    app.register_blueprint(main)

    # https://github.com/noirbizarre/flask-restplus/issues/712
    # Api.render_root returns 404; hence this hack
    # not sure why this is suddenly needed; it was working without it before
    app.add_url_rule('/api/v2/', endpoint='api.root_root_resource')

    api_bp = Blueprint('api', __name__, url_prefix='/api/v2')
    api.init_app(api_bp)
    api.add_namespace(models_ns)
    api.add_namespace(languages_ns)
    api.add_namespace(root_ns)
    app.register_blueprint(api_bp)

    @app.before_request
    def log_request_start():
        g.request_log_start = time.monotonic()
        g.request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
        app.logger.info(
            "API request started request_id=%s method=%s path=%s content_length=%s remote=%s",
            g.request_id,
            request.method,
            request.path,
            request.content_length,
            request.remote_addr,
        )

    @app.after_request
    def log_request_end(response):
        started = getattr(g, 'request_log_start', None)
        duration_ms = (
            (time.monotonic() - started) * 1000 if started is not None else -1
        )
        request_id = getattr(g, 'request_id', 'unknown')
        app.logger.info(
            "API request completed request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        response.headers.setdefault('X-Request-ID', request_id)
        return response

    return app
