"""Shared persistence helpers, not experiment-specific scoring shortcuts."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

def now():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def status(root,**fields):
    root=Path(root);local=root/'status.json'
    data=json.loads(local.read_text()) if local.exists() else {}
    data.update(fields,updated_at=now());save(local,data)
    target=root.parent.parent/'status/linkage.json'
    if target.parent.exists():save(target,data)
    return data
