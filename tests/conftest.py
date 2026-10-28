import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep tests independent of the developer's roles.yaml and role memory.
import os  # noqa: E402

os.environ["MMM_ROLES_FILE"] = str(Path(__file__).parent / "no_user_roles.yaml")
os.environ.pop("MMM_ROLE_MEMORY", None)
# Most test files are Indian insurance exports: they use the india and insurance keyword
# packs and suggest that calendar. tests/test_generic.py switches both off.
os.environ["MMM_KEYWORD_PACKS"] = "india,insurance"
os.environ["MMM_CALENDARS"] = str(Path(__file__).parent / "insurance_calendars.yaml")
