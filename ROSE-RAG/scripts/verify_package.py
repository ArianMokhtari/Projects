"""Verify the published archive; no solver, API or notebook execution."""
from pathlib import Path
import csv, gzip, hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(path, compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,"rb") if compressed else path.open("rb")) as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
count=0
with (ROOT/"package_manifest.csv").open(newline="",encoding="utf-8") as f:
    for row in csv.DictReader(f):
        p=ROOT/row["path"]
        if not p.is_file() or p.stat().st_size!=int(row["bytes"]) or digest(p)!=row["sha256"]:
            raise SystemExit("FAIL: "+row["path"])
        count+=1
with (ROOT/"asset_provenance.csv").open(newline="",encoding="utf-8") as f:
    assets=list(csv.DictReader(f))
for row in assets:
    p=ROOT/row["package_path"]
    if digest(p,p.suffix==".gz")!=row["sha256_uncompressed"]:
        raise SystemExit("FAIL decompressed input: "+row["package_path"])
print(f"PASS: {count} distributed files and {len(assets)} exposed source/input assets verified.")
