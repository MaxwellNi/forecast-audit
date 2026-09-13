"""Read-only SHA256 checks for this self-contained study export."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check_manifest():
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for name, expected in manifest['sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest entries must stay within the bundle')
        actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f'Bundle integrity mismatch: {name}')
    return len(manifest['sha256'])
