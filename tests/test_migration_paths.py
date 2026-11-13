import importlib.util
import json
import os
from pathlib import Path
import runpy
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask


ROOT = Path(__file__).resolve().parents[1]


class MigrationPathsTest(unittest.TestCase):
    def test_database_override_and_default(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'isolated.sqlite'
            with patch.dict(os.environ, {'LINDAT_DATABASE_PATH': str(target)}):
                spec = importlib.util.spec_from_file_location('isolated_db', ROOT / 'app/db/__init__.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            with Flask(__name__).app_context():
                connection = module.get_db()
                connection.execute('CREATE TABLE synthetic (value TEXT)')
                connection.execute("INSERT INTO synthetic VALUES ('test')")
                connection.commit()
                connection.close()
            with sqlite3.connect(target) as connection:
                self.assertEqual(connection.execute('SELECT value FROM synthetic').fetchone(), ('test',))
            with patch.dict(os.environ, {}, clear=True):
                default = runpy.run_path(str(ROOT / 'app/db/__init__.py'))
            # Inspect the default without connecting to its database.
            self.assertEqual(default['DATABASE'], str(ROOT / 'app/db/database.db'))

    def test_models_override_and_default(self):
        from app.models import Model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / 'model_settings.py'
            module.write_text((ROOT / 'app/model_settings.py').read_text())
            default = root / 'models.json'
            external = root / 'external.json'
            for path, name in [(default, 'default-synthetic'), (external, 'external-synthetic')]:
                path.write_text(json.dumps([{'model': name, 'source': ['en'], 'target': ['cs']}]))
            # Real graph/config loading, with no backend-specific initialization.
            with patch.object(Model, 'create', side_effect=Model):
                with patch.dict(os.environ, {}, clear=True):
                    result = runpy.run_path(str(module))
                self.assertEqual(result['models'].get_model_names(), ['default-synthetic'])
                with patch.dict(os.environ, {'LINDAT_MODELS_PATH': str(external)}):
                    result = runpy.run_path(str(module))
                self.assertEqual(result['models'].get_model_names(), ['external-synthetic'])
                with patch.dict(os.environ, {'LINDAT_MODELS_PATH': str(root / 'missing.json')}):
                    with self.assertRaises(FileNotFoundError):
                        runpy.run_path(str(module))


if __name__ == '__main__':
    unittest.main()
