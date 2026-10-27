import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep tests independent of the developer's roles.yaml and role memory.
import os  # noqa: E402

os.environ["MMM_ROLES_FILE"] = str(Path(__file__).parent / "no_user_roles.yaml")
os.environ.pop("MMM_ROLE_MEMORY", None)
