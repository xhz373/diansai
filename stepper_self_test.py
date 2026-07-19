import time


def _default_sleep_ms(delay_ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(delay_ms)
    else:
        time.sleep(delay_ms / 1000.0)


def _axis_ready(controller, axis_name):
    axis = getattr(controller, axis_name + "_axis", None)
    return bool(axis is not None and getattr(axis, "ready", False))


def _print_axis_status(controller, axis_name):
    axis = getattr(controller, axis_name + "_axis", None)
    if axis is None:
        print("[SelfTest] {} axis unavailable".format(axis_name.upper()))
        return

    if getattr(axis, "ready", False):
        print("[SelfTest] {} axis ready".format(axis_name.upper()))
    else:
        error = getattr(axis, "_init_error", "unknown init error")
        print("[SelfTest] {} axis failed: {}".format(axis_name.upper(), error))


def run_stepper_self_test(controller, move_error=2.0, move_ms=300,
                          pause_ms=150, sleep_ms=None):
    """Move each ready axis in both directions, then return to hold state."""
    sleep_ms = sleep_ms or _default_sleep_ms
    x_ready = _axis_ready(controller, "x")
    y_ready = _axis_ready(controller, "y")

    print("[SelfTest] stepper test start")
    _print_axis_status(controller, "x")
    _print_axis_status(controller, "y")

    if not x_ready and not y_ready:
        print("[SelfTest] FAIL: no stepper axis initialized")
        return False

    commands = (
        ("X command +", move_error, 0.0, x_ready),
        ("X command -", -move_error, 0.0, x_ready),
        ("Y command +", 0.0, move_error, y_ready),
        ("Y command -", 0.0, -move_error, y_ready),
    )

    try:
        for label, error_x, error_y, axis_ready in commands:
            if not axis_ready:
                continue
            print("[SelfTest] {}".format(label))
            controller.drive(error_x, error_y, allow_drive=True)
            sleep_ms(move_ms)
            controller.stop()
            sleep_ms(pause_ms)
    except Exception as e:
        print("[SelfTest] FAIL:", e)
        return False
    finally:
        try:
            controller.stop()
        except Exception:
            pass

    if x_ready and y_ready:
        print("[SelfTest] COMPLETE: verify X/Y moved and now resist hand rotation")
        return True

    print("[SelfTest] PARTIAL: verify the ready axis moved")
    return False
