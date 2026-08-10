"""Repo-root conftest.

Presence of this file makes pytest insert the repository root into
sys.path, so `src` and `scripts` are importable regardless of how pytest
is invoked (bare `pytest` vs `python -m pytest`).

Scripts under scripts/audit/ are executed directly rather than collected
by pytest, so they still require the repository root as cwd, or
PYTHONPATH=. — see README.
"""
