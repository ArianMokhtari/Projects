"""Compare a replay directory or SEND_ME ZIP to the author reference CSVs."""
from pathlib import Path
import csv,io,math,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2:raise SystemExit("Usage: python scripts/compare_outputs.py RUN_DIRECTORY_OR_ZIP")
source=Path(sys.argv[1]).expanduser();ref=ROOT/"expected_outputs/current"
z=zipfile.ZipFile(source) if source.is_file() else None
def get(name):return z.read(name) if z else (source/name).read_bytes()
TEXT_COLUMNS={"base_id","example_key","cluster_id","method","protocol","condition_id",
              "calibration_design","scoring_version","check","details","path","kind",
              "source_file","original_file","status","stratum","representation"}
def numeric_column(name, values):
    if name in TEXT_COLUMNS or name.endswith(("_id","_key","_hash","_sha256")):
        return False
    seen=False
    for value in values:
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
        seen=True
    return seen
exact=0;near=0
try:
    for p in sorted(ref.rglob("*.csv")):
        name=p.relative_to(ref).as_posix();a=p.read_bytes();b=get(name)
        if a==b:exact+=1;continue
        aa=list(csv.reader(io.StringIO(a.decode("utf-8-sig"))));bb=list(csv.reader(io.StringIO(b.decode("utf-8-sig"))))
        if len(aa)!=len(bb) or aa[0]!=bb[0]:raise SystemExit("FAIL shape/header: "+name)
        numeric=[numeric_column(name,[row[j] for row in aa[1:]]) for j,name in enumerate(aa[0])]
        for i,(rx,ry) in enumerate(zip(aa,bb)):
            if len(rx)!=len(ry):raise SystemExit("FAIL column count: "+name)
            for j,(sx,sy) in enumerate(zip(rx,ry)):
                if sx==sy:continue
                if i==0 or not numeric[j]:
                    raise SystemExit(f"FAIL text/identifier: {name}, CSV row {i+1}, column {aa[0][j]}")
                try:
                    x=float(sx);y=float(sy)
                    same=(math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
                except ValueError:same=False
                if not same:raise SystemExit(f"FAIL values: {name}, CSV row {i+1}")
        near+=1
finally:
    if z:z.close()
print(f"PASS: {exact} exact CSV matches; {near} tolerance-only matches.")
