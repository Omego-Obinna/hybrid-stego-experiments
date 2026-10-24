#!/usr/bin/env python3
"""Refresh release checksums after intentional edits; preserve source_manifest.json."""
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]
excluded={'.git','.venv','__pycache__','.crossref-sn-article','.crossref-Supplementary'}
files=[p for p in sorted(root.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(root).parts)
       and p.name!='SHA256SUMS' and p.suffix not in {'.aux','.log','.blg','.bbl','.out','.fls','.fdb_latexmk'}
       and str(p.relative_to(root)) not in {'manuscript/sn-article.pdf','manuscript/Supplementary.pdf'}]
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in files))
print(f'Updated SHA256SUMS for {len(files)} files; original source/package provenance is unchanged.')
