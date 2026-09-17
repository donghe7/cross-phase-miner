"""Run storage contracts in isolated PostgreSQL schemas, without resetting demo tables."""

import os
import unittest

from server.postgres_env import DEFAULT_FILE, database_url


def main():
    if not os.environ.get("CP_TEST_POSTGRES_URL"):
        os.environ["CP_TEST_POSTGRES_URL"] = database_url(DEFAULT_FILE)
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.server.test_store")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
