"""Verify the historical archive without extracting or executing its contents."""

import hashlib
import json
import tarfile
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[1] / "experiments/archive"
    manifest = json.loads((directory / "manifest.json").read_text())
    archive = directory / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if len(members) != len(manifest["files"]) or {m.name for m in members} != set(
            manifest["files"]
        ):
            raise ValueError("Archive file list mismatch")
        for name, expected in manifest["files"].items():
            content = handle.extractfile(name)
            if content is None or hashlib.sha256(content.read()).hexdigest() != expected:
                raise ValueError(f"Archived file checksum mismatch: {name}")
    print(f"Verified archive and {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
