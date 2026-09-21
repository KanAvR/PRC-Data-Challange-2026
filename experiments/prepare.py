"""Restore tracked experiment scripts and frozen v9 sources without overwrites."""
import io
import subprocess
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / 'experiments'
destination = root / 'output/experiments'
archive = subprocess.check_output(['git', 'archive', '4290754:src'], cwd=root)
pending = []
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    for member in tar.getmembers():
        if member.isfile():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe source archive path')
            pending.append((destination / 'v9_src' / relative, tar.extractfile(member).read()))
for path in source.rglob('*.py'):
    if path != Path(__file__).resolve():
        pending.append((destination / path.relative_to(source), path.read_bytes()))
for path, content in pending:
    if path.exists() and path.read_bytes() != content:
        raise SystemExit(f'Existing experiment differs; preserved: {path}')
for path, content in pending:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
print(f'Checked/restored {len(pending)} experiment and frozen-source files.')
