#!/usr/bin/env python3
"""Download only the corrected pu0 ROOT and signal HepMC inputs, without Docker."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


SOURCES = Path(__file__).resolve().parents[2] / "configs_development/higgs_portal_pilot/input_sources.json"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Wrong file size: {path}; expected {spec['bytes']} bytes")
    if spec.get("md5") and digest(path, "md5") != spec["md5"]:
        raise ValueError(f"Published signal MD5 does not match: {path}")


def download(directory, spec):
    path = directory / spec["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path, spec)
        print(f"Reusing verified {path}", flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec["bytes"]:
        raise ValueError(f"Oversized partial download: {partial}; retain it for diagnosis")
    if offset != spec["bytes"]:
        print(f"Downloading {spec['url']} ({spec['bytes'] / 1e9:.2f} GB), resuming at {offset}", flush=True)
        request = Request(spec["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=120) as response:
            if offset and response.status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected download range")
                mode = "ab"
            else:
                mode = "wb"
            received = offset if mode == "ab" else 0
            last_progress = received
            with partial.open(mode) as stream:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
                    received += len(chunk)
                    if received - last_progress >= 64 * 1024 * 1024:
                        print(f"  {100 * received / spec['bytes']:.1f}% ({received / 1e9:.2f} GB)", flush=True)
                        last_progress = received
    verify(partial, spec)
    partial.rename(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(SOURCES.read_text())
    provenance_path = args.output / "download_provenance.json"
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    records = {}
    for kind, spec in source["inputs"].items():
        path = download(args.output, spec)
        sha256 = digest(path)
        old_record = previous.get("inputs", {}).get(kind)
        if old_record and (old_record["url"] != spec["url"] or old_record["sha256"] != sha256):
            raise ValueError(f"Cached input differs from its saved download provenance: {path}")
        records[kind] = {**spec, "sha256": sha256}
    provenance = {**source, "inputs": records}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("Inputs ready. ROOT size is checked; signal MD5 is checked against the published manifest.")
    print("Local SHA256 checksums recorded; the archive does not supply a ROOT checksum.")


if __name__ == "__main__":
    main()
