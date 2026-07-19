"""
common_hw.py -- Shared hardware helpers for K230 CanMV gimbal project.

Covers FPIOA pin-mux, buttons, camera/display init, and draw utilities.
Import from this module instead of duplicating code across mode scripts.

Firmware: CanMV v1.8-0-gc2d1f5c (MicroPython e00a144)
"""

import gc
import os
import time
from machine import FPIOA, Pin

# ── Media modules (None on PC; real classes on K230) ──────────
try:
    from media.sensor import Sensor                                             # noqa: F401
except ImportError:
    Sensor = None                                                               # type: ignore

try:
    from media.media import MediaManager                                        # noqa: F401
except ImportError:
    MediaManager = None                                                         # type: ignore

try:
    from media.display import Display                                           # noqa: F401
except ImportError:
    Display = None                                                              # type: ignore


# ═══════════════════════════════════════════════════════════════
#  FPIOA pin-mux helpers
# ═══════════════════════════════════════════════════════════════

def map_gpio(fpioa, board_pin, gpio_num):
    """Configure FPIOA to route *board_pin* to a GPIO function.

    Tries multiple naming conventions to work across CanMV firmware builds.
    """
    if not hasattr(fpioa, "set_function"):
        return
    for func_name in (
        "GPIO{}_FUNC".format(gpio_num),
        "GPIO{}".format(gpio_num),
        "GPIOHS{}".format(gpio_num),
        "GPIOHS{}_FUNC".format(gpio_num),
    ):
        func = getattr(fpioa, func_name, None)
        if func is not None:
            fpioa.set_function(board_pin, func)
            return


def map_pwm(fpioa, board_pin, pwm_channel=None):
    """Configure FPIOA to route *board_pin* to a PWM function.

    K230 CanMV firmware (v1.8) does **not** auto-configure FPIOA for PWM,
    so this must be called before creating a ``machine.PWM`` on the pin.

    If *pwm_channel* is given (0-5), that channel is tried first; otherwise
    all channels PWM0-PWM5 are probed.
    """
    if not hasattr(fpioa, "set_function"):
        return False

    channels_to_try = []
    if pwm_channel is not None:
        channels_to_try.append(int(pwm_channel))
    channels_to_try.extend([ch for ch in range(6) if ch not in channels_to_try])

    for ch in channels_to_try:
        # CanMV v1.8 uses "PWM0" / "PWM1" style (no _FUNC suffix).
        for func_name in ("PWM{}_FUNC".format(ch), "PWM{}".format(ch)):
            func = getattr(fpioa, func_name, None)
            if func is not None:
                try:
                    fpioa.set_function(board_pin, func)
                    return True
                except Exception:
                    pass
    return False


def pin_pull_up():
    """Return the internal pull-up constant, or ``None`` if unavailable."""
    for name in ("PULL_UP", "PULLUP", "PULL_UP_ENABLE"):
        value = getattr(Pin, name, None)
        if value is not None:
            return value
    return None


def pin_out():
    """Return ``Pin.OUT`` mode constant."""
    for name in ("OUT",):
        value = getattr(Pin, name, None)
        if value is not None:
            return value
    raise AttributeError("Pin.OUT not available")


# ═══════════════════════════════════════════════════════════════
#  Button classes
# ═══════════════════════════════════════════════════════════════

class DebouncedButton:
    """Single debounced, latched hardware button."""

    def __init__(self, board_pin, gpio_num, debounce_ms=35):
        self.pin = None
        self.ready = False
        self.latched = False
        self.last_raw_value = 1
        self.stable_value = 1
        self.last_change_ms = 0
        self.debounce_ms = int(debounce_ms)
        self._init(board_pin, gpio_num)

    def _init(self, board_pin, gpio_num):
        try:
            fpioa = FPIOA()
            map_gpio(fpioa, board_pin, gpio_num)
            pull_up = pin_pull_up()
            try:
                if pull_up is None:
                    self.pin = Pin(gpio_num, Pin.IN)
                else:
                    self.pin = Pin(gpio_num, Pin.IN, pull_up)
            except Exception:
                if pull_up is None:
                    self.pin = Pin(board_pin, Pin.IN)
                else:
                    self.pin = Pin(board_pin, Pin.IN, pull_up)
            self.ready = self.pin is not None
        except Exception as e:
            print("[Button] init failed:", e)
            self.pin = None
            self.ready = False

    def poll_pressed(self):
        if self.latched or self.pin is None:
            return self.latched
        now = time.ticks_ms()
        try:
            raw_value = self.pin.value()
        except Exception:
            return False
        if raw_value != self.last_raw_value:
            self.last_raw_value = raw_value
            self.last_change_ms = now
            return False
        if time.ticks_diff(now, self.last_change_ms) < self.debounce_ms:
            return False
        if raw_value != self.stable_value:
            self.stable_value = raw_value
            if self.stable_value == 0:
                self.latched = True
                return True
        return False

    def reset(self):
        self.latched = False


StartButton = DebouncedButton


class MultiButton:
    """Four-button input system for calibration mode."""

    def __init__(self, button_configs, debounce_ms=35,
                 long_press_ms=700, repeat_delay_ms=320, repeat_ms=70):
        self.buttons = {}
        self._error = ""
        self._debounce_ms = debounce_ms
        self._long_ms = long_press_ms
        self._repeat_delay_ms = repeat_delay_ms
        self._repeat_ms = repeat_ms
        self._init(button_configs)

    def _init(self, button_configs):
        try:
            fpioa = FPIOA()
            pull_up = pin_pull_up()
        except Exception as e:
            self._error = str(e)
            print("[MultiButton] init failed:", e)
            return

        for name, board_pin, gpio_num in button_configs:
            try:
                map_gpio(fpioa, board_pin, gpio_num)
                if pull_up is None:
                    try:
                        pin = Pin(gpio_num, Pin.IN)
                    except Exception:
                        pin = Pin(board_pin, Pin.IN)
                else:
                    try:
                        pin = Pin(gpio_num, Pin.IN, pull_up)
                    except Exception:
                        pin = Pin(board_pin, Pin.IN, pull_up)
                self.buttons[name] = {
                    "pin": pin,
                    "pressed": False,
                    "stable_value": 1,
                    "last_raw_value": 1,
                    "last_change_ms": 0,
                    "down_ms": 0,
                    "last_repeat_ms": 0,
                    "long_fired": False,
                }
            except Exception as e:
                self._error += "{}:{}; ".format(name, e)

        if self.buttons:
            print("[Keys] ready:", ",".join(self.buttons.keys()))
        else:
            print("[Keys] init failed")

    def poll_events(self):
        events = []
        now = time.ticks_ms()
        for name in list(self.buttons.keys()):
            st = self.buttons[name]
            try:
                raw = st["pin"].value()
            except Exception:
                continue

            if raw != st["last_raw_value"]:
                st["last_raw_value"] = raw
                st["last_change_ms"] = now
                continue
            if time.ticks_diff(now, st["last_change_ms"]) < self._debounce_ms:
                continue
            if raw != st["stable_value"]:
                st["stable_value"] = raw

            is_down = (st["stable_value"] == 0)

            if is_down and not st["pressed"]:
                st["pressed"] = True
                st["down_ms"] = now
                st["last_repeat_ms"] = now
                st["long_fired"] = False
                events.append((name, "press"))
            elif is_down and st["pressed"]:
                hold_ms = time.ticks_diff(now, st["down_ms"])
                if (not st["long_fired"]) and hold_ms >= self._long_ms:
                    st["long_fired"] = True
                    st["last_repeat_ms"] = now
                    events.append((name, "long"))
                elif name in ("KEY3", "KEY4") and hold_ms >= self._repeat_delay_ms:
                    if time.ticks_diff(now, st["last_repeat_ms"]) >= self._repeat_ms:
                        st["last_repeat_ms"] = now
                        events.append((name, "repeat"))
            elif (not is_down) and st["pressed"]:
                st["pressed"] = False
                if not st["long_fired"]:
                    events.append((name, "short"))
        return events

    def is_ready(self):
        return len(self.buttons) == 4


# ═══════════════════════════════════════════════════════════════
#  Drawing helper
# ═══════════════════════════════════════════════════════════════

def draw_text(img, x, y, text, color=(255, 255, 255), scale=1):
    text = str(text)
    if hasattr(img, "draw_string_advanced"):
        img.draw_string_advanced(x, y, max(16, 16 * scale), text, color=color)
    else:
        img.draw_string(x, y, text, color=color, scale=scale)


# ═══════════════════════════════════════════════════════════════
#  Camera / display helpers
# ═══════════════════════════════════════════════════════════════

_ACTIVE_CHN = 0  # 彻底锁死在 chn0


def _chn_name(chn):
    if chn == 2:
        return "chn2"
    return "chn1" if chn == 1 else "chn0"


def camera_configure(sensor, width, height, hmirror=True, vflip=True, chn=2):
    """Reset sensor and configure resolution / pixel format (OV5647, default chn=2)."""
    sensor.reset()
    try:
        sensor.set_hmirror(hmirror)
    except Exception:
        pass
    try:
        sensor.set_vflip(vflip)
    except Exception:
        pass
    if chn in (1, 2):
        # ISP 输出通道：chn=0 作为 sensor 基础输出，chn=1/2 为缩放 RGB 输出
        sensor.set_framesize(width=width, height=height)
        sensor.set_pixformat(Sensor.RGB565)
        sensor.set_framesize(width=width, height=height, chn=chn)
        sensor.set_pixformat(Sensor.RGB565, chn=chn)
    else:
        sensor.set_framesize(width=width, height=height)
        sensor.set_pixformat(Sensor.RGB565)


def camera_init(camera_id=2):
    """Create and return a ``Sensor`` for *camera_id*."""
    try:
        s = Sensor(id=camera_id)
    except OSError as e:
        if "already inited" not in str(e):
            raise
        Sensor.deinit()
        time.sleep_ms(20)
        s = Sensor(id=camera_id)
    return s


def camera_start(sensor, camera_id=2, width=400, height=300,
                 hmirror=True, vflip=True, allow_fallback=False,
                 retry_count=2, retry_delay_ms=10,
                 settle_ms=28, settle_step_ms=18,
                 warmup_frames=2):
    """Start sensor stream with auto-retry.  Sets module-level ``_ACTIVE_CHN``."""
    global _ACTIVE_CHN

    # 板载 OV5647 默认使用 chn=2（ISP 输出），回退到 chn=1、chn=0
    channels = (2, 1, 0) if allow_fallback else (2,)

    last_error = None
    for chn in channels:
        for attempt in range(retry_count):
            os.exitpoint()
            try:
                # 1. 先初始化媒体管理器
                MediaManager.init()
                
                # 2. 【核心修改】将配置移动到这里！确保在 MediaManager.init() 之后执行 sensor.reset()
                try:
                    camera_configure(sensor, width, height, hmirror, vflip, chn=chn)
                except Exception as ce:
                    print("[Sensor] config failed inside loop:", ce)

                # 3. 此时状态正确，可以安全启动
                sensor.run()
                _settle = settle_ms + attempt * settle_step_ms
                time.sleep_ms(_settle)
                for _ in range(warmup_frames):
                    os.exitpoint()
                    try:
                        _snap(sensor, chn)
                        _ACTIVE_CHN = chn
                        return True
                    except Exception as e:
                        last_error = e
                        time.sleep_ms(3)
                last_error = RuntimeError(
                    "camera {} no warmup on {}".format(
                        camera_id, _chn_name(chn)))
            except Exception as e:
                last_error = e

            try:
                sensor.stop()
            except Exception:
                pass
            try:
                MediaManager.deinit()
            except Exception:
                pass
            gc.collect()
            time.sleep_ms(retry_delay_ms)

    if last_error:
        raise last_error
    raise RuntimeError("camera start failed")


def _snap(sensor, chn):
    # 使用活跃通道，失败依次回退 chn=2 → chn=1 → chn=0
    fallback_order = [c for c in (2, 1, 0) if c != chn]
    try:
        return sensor.snapshot(chn=chn)
    except Exception:
        for fb in fallback_order:
            try:
                return sensor.snapshot(chn=fb)
            except Exception:
                pass
        return sensor.snapshot()


def camera_snapshot(sensor, retry_count=3, retry_delay_ms=3):
    """Take one frame with automatic retry."""
    last_error = None
    for _ in range(retry_count):
        os.exitpoint()
        try:
            return _snap(sensor, _ACTIVE_CHN)
        except RuntimeError as e:
            last_error = e
            time.sleep_ms(retry_delay_ms)
    raise last_error


def camera_restart(sensor, camera_id=2, width=400, height=300,
                   hmirror=True, vflip=True, **kw):
    """Full camera restart: stop → deinit → init → start."""
    os.exitpoint()
    print("[Sensor] restarting...")
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    try:
        Sensor.deinit()
    except Exception:
        pass
    gc.collect()
    time.sleep_ms(20)
    s = camera_init(camera_id)
    camera_start(s, camera_id=camera_id, width=width, height=height,
                 hmirror=hmirror, vflip=vflip, **kw)
    print("[Sensor] restart done")
    return s


def camera_deinit(sensor):
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    try:
        MediaManager.deinit()
    except Exception:
        pass
    try:
        Sensor.deinit()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  Display init
# ═══════════════════════════════════════════════════════════════

def display_init(width=400, height=300, use_st7701_fallback=True):
    try:
        Display.init(Display.VIRT, width=width, height=height,
                     fps=100, to_ide=True)
        print("[Display] VIRT preview {}x{}".format(width, height))
    except Exception as e:
        print("[Display] VIRT failed:", e)
        if use_st7701_fallback:
            Display.init(Display.ST7701, to_ide=True)
            print("[Display] ST7701 preview")
        else:
            raise


def display_init_board():
    Display.init(Display.ST7701, to_ide=True)
    print("[Display] ST7701 key preview")