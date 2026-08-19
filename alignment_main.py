"""Launch the standalone Phase 1 Raman alignment monitor."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from alignment.profile import ORCA_QUEST2_SI_520_PROFILE
from alignment.window import AlignmentWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = AlignmentWindow(ORCA_QUEST2_SI_520_PROFILE)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
