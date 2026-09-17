"""Initialize local Compose credentials and form an escaped connection URL.

The generated file is ignored by git and is never sourced as shell code.
Both Compose and this helper honor CP_PG_* environment overrides.
"""

import argparse
import os
import re
import secrets
from pathlib import Path
from urllib.parse import quote

DEFAULT_FILE = Path(__file__).with_name(".env.postgres")
KEYS = ("CP_PG_USER", "CP_PG_DATABASE", "CP_PG_PASSWORD", "CP_PG_PORT")


def initialize(path: Path) -> None:
    if path.exists():
        return
    values = dict(
        CP_PG_USER="crossphase",
        CP_PG_DATABASE="crossphase",
        CP_PG_PASSWORD=secrets.token_hex(24),
        CP_PG_PORT="5432",
    )
    values.update({key: os.environ[key] for key in KEYS if key in os.environ})
    validate(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as stream:
        stream.write("# Local PostgreSQL credentials. Keep this file private.\n")
        stream.writelines(f"{key}={values[key]}\n" for key in KEYS)


def validate(values: dict) -> None:
    for key in KEYS:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", values.get(key, "")):
            raise ValueError(f"{key} must contain only letters, digits, _ or -")
    if not values["CP_PG_PORT"].isdigit() or not 1 <= int(values["CP_PG_PORT"]) <= 65535:
        raise ValueError("CP_PG_PORT must be between 1 and 65535")


def read_settings(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    values.update({key: os.environ[key] for key in KEYS if key in os.environ})
    validate(values)
    return values


def database_url(path: Path) -> str:
    values = read_settings(path)
    user = quote(values["CP_PG_USER"], safe="")
    password = quote(values["CP_PG_PASSWORD"], safe="")
    database = quote(values["CP_PG_DATABASE"], safe="")
    return f"postgresql://{user}:{password}@127.0.0.1:{values['CP_PG_PORT']}/{database}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "url"])
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.file)
            read_settings(args.file)
        else:
            print(database_url(args.file))
    except (OSError, ValueError) as error:
        parser.exit(1, f"PostgreSQL configuration error: {error}\n")


if __name__ == "__main__":
    main()
