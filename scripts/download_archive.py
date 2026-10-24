"""Download the Halterman & Keith replication archive (doi:10.7910/DVN/NUWHQP)
from Harvard Dataverse and verify every file against its published MD5.
LoRA adapter weights (adapter_model.safetensors, ~6.6 GB, only needed for the
fine-tuning level we do not run) are skipped unless --with-adapters is given."""
import argparse, hashlib, json, os, sys, urllib.request

API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/NUWHQP"
VERSION = "1.2"  # version used for this replication
# Dataverse rejects urllib's default User-Agent with 403
urllib.request.install_opener(urllib.request.build_opener())
urllib.request._opener.addheaders = [("User-Agent", "Mozilla/5.0 (hk2025-replication)")]

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="archive")
    ap.add_argument("--with-adapters", action="store_true")
    a = ap.parse_args()
    meta = json.load(urllib.request.urlopen(f"{API}/datasets/:persistentId/versions/{VERSION}?persistentId={DOI}"))
    files = meta["data"]["files"]
    bad = []
    for f in files:
        df = f["dataFile"]
        if df["filename"] == ".DS_Store":
            continue
        if df["filename"].endswith(".safetensors") and not a.with_adapters:
            print("skip", df["filename"]); continue
        d = os.path.join(a.out, f.get("directoryLabel", ""))
        os.makedirs(d, exist_ok=True)
        # Dataverse renames ingested CSVs to .tab; the authors' code expects the
        # original names, and format=original returns the original bytes
        p = os.path.join(d, df.get("originalFileName", df["filename"]))
        want = df["checksum"]["value"]
        if not (os.path.exists(p) and md5(p) == want):
            # format=original returns the uploaded bytes for ingested tabular files
            q = "?format=original" if df.get("originalFileFormat") else ""
            urllib.request.urlretrieve(f"{API}/access/datafile/{df['id']}{q}", p)
        ok = md5(p) == want
        print("OK " if ok else "BAD", p)
        if not ok: bad.append(p)
    print(f"{len(bad)} checksum failures"); sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
