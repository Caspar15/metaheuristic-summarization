"""Gate: the nltk version pin and the vendored punkt_tab data must move
together, and this must be enforced, not just documented.

src/baselines/centrality.py inserts vendor/nltk_punkt_tab/ at the FRONT of
nltk.data.path (see that module's own comment for why -- the offline
compute cluster runs `python -m src.baselines.cli` directly, never through
conftest.py, so the registration has to live in production code, not a test
fixture). That front-insertion priority is exactly what makes a version
mismatch dangerous: nltk 3.8.x reads the classic punkt pickle directly,
while nltk 3.9+'s switch_punkt shim requires punkt_tab instead (see
docs/research/COMPUTE_ENVIRONMENT.md's "nltk 版本 pin" section). If nltk is
ever upgraded without regenerating vendor/nltk_punkt_tab/ for the new
version, nothing here would notice on its own -- the vendored data still
sits at the front of the search path and would keep "winning" silently,
whether or not it is still the right data for whatever nltk version ended
up installed. These two tests together are the gate that catches that:

  1. the installed nltk is exactly the version requirements.txt pins (not
     read as a second hardcoded literal here -- parsed from requirements.txt
     itself, so this test can't silently drift out of sync with the actual
     pin the way two independently-maintained copies of a version number
     eventually do), and
  2. the vendored punkt_tab resource still resolves at all, and resolves to
     the vendored copy specifically (not some other punkt_tab this
     particular machine happens to already have cached).

Neither check implies the other -- nltk could still match while the
vendored data went stale some other way (e.g. hand-edited), or the vendored
data could still resolve while nltk silently drifted -- so both must pass
independently. Upgrading nltk means: bump the pin, regenerate
vendor/nltk_punkt_tab/ per its own README, then re-run these two tests
before trusting anything else in this module.
"""

import re
from pathlib import Path

import nltk

import src.baselines.centrality  # noqa: F401 -- triggers import-time nltk.data.path registration


def _pinned_nltk_version() -> str:
    requirements_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    text = requirements_path.read_text(encoding="utf-8")
    match = re.search(r"^nltk==([0-9.]+)\s*$", text, re.MULTILINE)
    assert match is not None, (
        "requirements.txt must pin nltk with an exact '==' version, not a "
        "floor -- see docs/research/COMPUTE_ENVIRONMENT.md's 'nltk 版本 pin' "
        "section for why a floor (e.g. '>=3.8.1') is not sufficient here"
    )
    return match.group(1)


def test_installed_nltk_matches_the_exact_pin_in_requirements_txt():
    """Exact match, not a range check: this is the version the vendored
    punkt_tab data in vendor/nltk_punkt_tab/ was generated for and verified
    against (see that directory's README and the CI traceback recorded in
    docs/research/COMPUTE_ENVIRONMENT.md). A newer or older nltk silently
    installed (e.g. via a lax requirements resolution elsewhere) would not
    itself raise anywhere else in this codebase -- this is the check that
    would catch it."""

    assert nltk.__version__ == _pinned_nltk_version()


def test_vendored_punkt_tab_still_resolves_under_the_pinned_nltk():
    """The companion half of the gate: confirms the vendored resource is
    still loadable at all under whatever nltk is actually installed right
    now, and that it resolves to vendor/nltk_punkt_tab/ specifically --
    not some other punkt_tab this machine happens to already have cached
    (which the front-insertion priority in centrality.py is supposed to
    prevent, but this test is what actually verifies it did)."""

    resolved = nltk.data.find("tokenizers/punkt_tab/english/")
    assert "vendor/nltk_punkt_tab" in str(resolved)
