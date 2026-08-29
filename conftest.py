# Intentionally empty.
#
# Its presence at the project root is what matters: pytest uses the location
# of the topmost conftest.py to determine the "rootdir" it inserts onto
# sys.path before collecting tests. Without this file, pytest instead
# inserts the tests/ directory itself (since it has no __init__.py), which
# means `from app.database import Base` fails with
# `ModuleNotFoundError: No module named 'app'` even though app/ sits right
# next to tests/ in the project.
