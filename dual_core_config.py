from control_backends import BACKEND_LOCAL


# Default keeps the current repo runnable on a single core.
# Switch to "sharefs_ipc" when wiring the little-core Python side to the big-core RT side.
CONTROL_BACKEND = BACKEND_LOCAL
