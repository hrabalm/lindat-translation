# Deployment paths

Set `LINDAT_DATABASE_PATH` to an absolute SQLite path and `LINDAT_MODELS_PATH` to an
absolute model JSON path before importing the application. Unset variables preserve
the package-relative `app/db/database.db` and `app/models.json` defaults. These are
environment overrides, independent of Flask `LOCAL_SETTINGS`.

Preflight must verify that the configured database exists and has the expected
schema: SQLite can create an empty database when the path is missing. Its parent
must permit SQLite sidecar writes. Invalid model paths fail instead of falling back.
No schema or provisioning behavior is changed.

The existing `TIKAL_PATH` accepts a separate complete Okapi distribution and requires
a trailing slash. Configure `ALIGNER_SLOT_DIR` explicitly and coordinate permissions
and concurrency when two service identities overlap.

Run the isolated path tests with `.venv/bin/python -m unittest discover -s tests`.
Staging requires a separate online SQLite backup, never the live production file.
