"""Backwards-compatible entry point for the Jarvis Code Assistant web UI.

Run with ``python code_ui.py`` (serves :8500 by default). The real
implementation lives in the ``code_ui/`` package — see
``code_ui/lifecycle.py`` and the package docstring.
"""

from code_ui.lifecycle import main

if __name__ == "__main__":
    main()
