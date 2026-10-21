"""Generate the synthetic demo data. Thin wrapper over `python -m sqlagent.synthetic`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlagent.synthetic import main  # noqa: E402

if __name__ == "__main__":
    main()
