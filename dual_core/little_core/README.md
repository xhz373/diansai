# Little-Core Side

The existing Python mode files already act as the little-core side:

- `stand_aiming.py`
- `move_aiming.py`
- `circle_mode.py`

Backend selection:

- Edit `dual_core_config.py`
- `CONTROL_BACKEND = "local"` keeps local PID/PWM control on the same runtime
- `CONTROL_BACKEND = "sharefs_ipc"` switches the vision side to packet publishing

When `sharefs_ipc` is enabled, the Python side publishes the latest control packet to:

```text
/sharefs/k230_dual_core/vision_control_latest.json
```

This is intended for early dual-core bring-up only.
