"""Allow ``python -m hud`` to launch the desktop HUD."""

from .app import main

raise SystemExit(main())
