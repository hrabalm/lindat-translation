import atexit
import os
import shutil
import tempfile


def _create_upload_folder(base_dir=None):
    identity = getattr(os, "getuid", os.getpid)()
    return tempfile.mkdtemp(
        prefix=f"lindat-translation-{identity}-",
        dir=base_dir,
    )


def _cleanup_upload_folder(path, owner_pid):
    # With Gunicorn preloading, workers inherit the master's atexit handler.
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)

BOOTSTRAP_SERVE_LOCAL = True
ERROR_404_HELP = False
RESTX_MASK_SWAGGER = False

# These should match with the appropriate constants in the frontend
# Maximum uploaded file size length is checked by Flask: https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/#improving-uploads
MAX_CONTENT_LENGTH = 20 * 1024 * 1024
# Maximum text length inside the uploaded file
MAX_TEXT_LENGTH = 100 * 1024

BATCH_SIZE = 20 #1000
MARIAN_BATCH_SIZE = 16
SENT_LEN_LIMIT = 500
#CSRF prevention
SECRET_KEY = (os.environ.get('SECRET_KEY') or
              b'\x0c\x11{\xd3\x11$\xeeel\xa6\xfb\x1d~\xfd\xb3\x9d\x11\x00\xfb4\xd64\xd4\xe0')
#DEFAULT_SERVER = 'localhost:9000'
#DEFAULT_SERVER = '10.10.51.30:9000'
#T2T_TRANSFORMER2 = '10.10.51.31:9000'
#MARIAN = '10.10.51.50'
DEFAULT_SERVER = '10.10.51.71:9000'
GPU2 = '10.10.51.72:9000'
GPU3 = '10.10.51.73:9000'
ENCS_LOAD_BALANCED = '10.10.51.71:9000'
CSEN_LOAD_BALANCED = '10.10.51.72:9000'
DOCLVL_LOAD_BALANCED = '10.10.51.76:9000'

_UPLOAD_FOLDER_OWNER_PID = os.getpid()
UPLOAD_FOLDER = _create_upload_folder()
atexit.register(
    _cleanup_upload_folder,
    UPLOAD_FOLDER,
    _UPLOAD_FOLDER_OWNER_PID,
)

# These should match with the appropriate constants in the frontend
# inxml, innopxml are our custom XML formats for ČTK that treasts all tags/all tags except <p> as inline tags
ALLOWED_EXTENSIONS = {'inxml', 'innopxml', 'txt', 'xml', 'html', 'htm', 'docx', 'odt', 'pptx', 'odp', 'xlsx', 'ods', 'pdf', 'srt', 'md', 'xlf', 'json', 'yaml'}

TIKAL_PATH = os.environ.get('TIKAL_PATH', '/home/jon/okapi-apps/')
MODEL_SERVER = os.environ.get('MODEL_SERVER')
ALIGNER_BASE_URL = os.environ.get('ALIGNER_BASE_URL', 'https://lindat.cz/services/text-aligner')
ALIGNER_CONNECT_TIMEOUT = float(os.environ.get('ALIGNER_CONNECT_TIMEOUT', '10'))
ALIGNER_READ_TIMEOUT = float(os.environ.get('ALIGNER_READ_TIMEOUT', '120'))
ALIGNER_MAX_ATTEMPTS = int(os.environ.get('ALIGNER_MAX_ATTEMPTS', '5'))
ALIGNER_RETRY_DEADLINE = float(os.environ.get('ALIGNER_RETRY_DEADLINE', '180'))
ALIGNER_MAX_IN_FLIGHT = int(os.environ.get('ALIGNER_MAX_IN_FLIGHT', '4'))
ALIGNER_QUEUE_TIMEOUT = float(os.environ.get('ALIGNER_QUEUE_TIMEOUT', '60'))
# Stable across workers/restarts; unlike UPLOAD_FOLDER this must not be removed
# on worker shutdown. All workers for an endpoint must use the same limit/path.
ALIGNER_SLOT_DIR = os.environ.get('ALIGNER_SLOT_DIR', os.path.join(
    tempfile.gettempdir(), f'lindat-alignment-{getattr(os, "getuid", os.getpid)()}'))
ALIGNER_FAILURE_POLICY = os.environ.get('ALIGNER_FAILURE_POLICY', 'recover')
FRAUS_V2_FORCE_SENTENCE_LEVEL = os.environ.get(
    'FRAUS_V2_FORCE_SENTENCE_LEVEL', 'true'
).lower() in {'1', 'true', 'yes'}
FRAUS_V2_MAX_SEGMENT_TOKENS = int(
    os.environ.get('FRAUS_V2_MAX_SEGMENT_TOKENS', '100')
)
