from flask import make_response, render_template
from flask_restx import Api

from app.models.llm_errors import LLMBackendError

# TODO terms, etc.
api = Api(version='2.0', title='LINDAT Translation API', default_mediatype=None,
          contact_email='lindat-technical@ufal.mff.cuni.cz', doc='/doc')


@api.errorhandler(LLMBackendError)
def handle_llm_backend_error(error):
    return {"message": error.public_message}, error.status_code


@api.representation('text/plain')
def output_text(data, code, headers=None):
    message = data.get('message', '') if isinstance(data, dict) else str(data)
    response = make_response(message, code)
    response.headers.extend(headers or {})
    return response


@api.documentation
def custom_ui():
    return render_template('swagger-ui.html', title=api.title,
                           specs_url=api.specs_url)
