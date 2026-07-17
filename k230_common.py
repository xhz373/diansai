import json
import os
import time


CALIB_FILE = "/sdcard/app/aiming_calib.json"

# Default two-axis stepper wiring. Adjust these board pins to match your driver.
# Each axis assumes a STEP/DIR/EN driver such as A4988/DRV8825/TB6600.
DEFAULT_STEPPER_AXES = {
    "x": {
        "name": "X",
        "step_board_pin": 42,
        "dir_board_pin": 43,
        "dir_gpio_num": 43,
        "enable_board_pin": 44,
        "enable_gpio_num": 44,
        "command_sign": 1,
        "dir_invert": False,
        "enable_active_low": True,
        "hold_enabled": True,
        "step_duty": 50,
        "min_freq": 120,
        "max_freq": 1800,
        "ramp_hz_per_s": 3200,
        "deadband": 0.0,
        "error_full_scale": 80.0,
        "pid_kp": 24.0,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "integral_limit": 200.0,
        "integral_active_error": 80.0,
        "derivative_alpha": 0.25,
    },
    "y": {
        "name": "Y",
        "step_board_pin": 8,
        "dir_board_pin": 7,
        "dir_gpio_num": 7,
        "enable_board_pin": 6,
        "enable_gpio_num": 6,
        "command_sign": 1,
        "dir_invert": False,
        "enable_active_low": True,
        "hold_enabled": True,
        "step_duty": 50,
        "min_freq": 120,
        "max_freq": 1800,
        "ramp_hz_per_s": 3200,
        "deadband": 0.0,
        "error_full_scale": 80.0,
        "pid_kp": 24.0,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "integral_limit": 200.0,
        "integral_active_error": 80.0,
        "derivative_alpha": 0.25,
    },
}


def _ensure_parent_dir(path):
    parts = path.split("/")[:-1]
    current = ""
    for part in parts:
        if not part:
            current = "/"
            continue
        if current == "/":
            current = "/" + part
        else:
            current = current + "/" + part
        try:
            os.stat(current)
        except OSError:
            try:
                os.mkdir(current)
            except OSError:
                pass


def load_calibration(default_red, default_black, default_violet, default_bright=None):
    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
        red = tuple(data.get("red_threshold", default_red))
        black = tuple(data.get("black_threshold", default_black))
        violet = tuple(data.get("violet_threshold", default_violet))
        if default_bright is None:
            bright = None
        else:
            bright = tuple(data.get("bright_threshold", default_bright))
        return True, red, black, violet, bright
    except Exception as e:
        print("[Calib] load failed:", e)
        return False, tuple(default_red), tuple(default_black), tuple(default_violet), default_bright


def save_calibration(red_threshold, black_threshold, violet_threshold, bright_threshold=None):
    payload = {
        "red_threshold": list(red_threshold),
        "black_threshold": list(black_threshold),
        "violet_threshold": list(violet_threshold),
    }
    if bright_threshold is not None:
        payload["bright_threshold"] = list(bright_threshold)

    try:
        _ensure_parent_dir(CALIB_FILE)
        with open(CALIB_FILE, "w") as f:
            json.dump(payload, f)
        print("[Calib] saved:", CALIB_FILE)
        return True
    except Exception as e:
        print("[Calib] save failed:", e)
        return False


def _merge_axis_config(defaults, overrides):
    merged = {}
    for key in defaults:
        merged[key] = defaults[key]
    if overrides:
        for key in overrides:
            merged[key] = overrides[key]
    return merged


def _pin_output_mode(pin_cls):
    for name in ("OUT",):
        value = getattr(pin_cls, name, None)
        if value is not None:
            return value
    raise AttributeError("Pin.OUT is not available")


def _map_board_pin_to_gpio(fpioa, board_pin, gpio_num):
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


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


class StepperAxis:
    def __init__(self, config):
        self.name = config.get("name", "?")
        self.step_board_pin = config.get("step_board_pin")
        self.dir_board_pin = config.get("dir_board_pin")
        self.dir_gpio_num = config.get("dir_gpio_num", self.dir_board_pin)
        self.enable_board_pin = config.get("enable_board_pin")
        self.enable_gpio_num = config.get("enable_gpio_num", self.enable_board_pin)
        self.command_sign = config.get("command_sign", 1)
        self.dir_invert = bool(config.get("dir_invert", False))
        self.enable_active_low = bool(config.get("enable_active_low", True))
        self.hold_enabled = bool(config.get("hold_enabled", True))
        self.step_duty = int(config.get("step_duty", 50))
        self.min_freq = int(config.get("min_freq", 120))
        self.max_freq = int(config.get("max_freq", 1800))
        self.ramp_hz_per_s = float(config.get("ramp_hz_per_s", 3200))
        self.deadband = float(config.get("deadband", 0.0))
        self.error_full_scale = max(float(config.get("error_full_scale", 80.0)), self.deadband + 1e-6)
        self.pid_kp = float(config.get("pid_kp", 24.0))
        self.pid_ki = float(config.get("pid_ki", 0.0))
        self.pid_kd = float(config.get("pid_kd", 0.0))
        self.integral_limit = abs(float(config.get("integral_limit", self.error_full_scale * 4.0)))
        self.integral_active_error = abs(
            float(config.get("integral_active_error", self.error_full_scale))
        )
        self.derivative_alpha = _clamp(
            float(config.get("derivative_alpha", 0.25)),
            0.0,
            0.98,
        )
        self._dir_pin = None
        self._enable_pin = None
        self._pwm = None
        self._current_freq = 0.0
        self._last_update_ms = time.ticks_ms()
        self._pid_ready = False
        self._last_error = 0.0
        self._integral = 0.0
        self._derivative = 0.0
        self._last_output = 0.0
        self.ready = False
        self._init_failed = False
        self._init_error = ""
        self._init_hw()

    def _init_hw(self):
        if self.step_board_pin is None or self.dir_board_pin is None:
            self._init_failed = True
            self._init_error = "missing step/dir pin"
            print("[Stepper:{}] disabled: {}".format(self.name, self._init_error))
            return

        try:
            machine = __import__("machine")
            Pin = getattr(machine, "Pin")
            FPIOA = getattr(machine, "FPIOA", None)
            if FPIOA is not None:
                try:
                    _map_board_pin_to_gpio(FPIOA(), self.dir_board_pin, self.dir_gpio_num)
                    if self.enable_board_pin is not None and self.enable_gpio_num is not None:
                        _map_board_pin_to_gpio(FPIOA(), self.enable_board_pin, self.enable_gpio_num)
                except Exception:
                    pass
            output_mode = _pin_output_mode(Pin)
            try:
                self._dir_pin = Pin(self.dir_gpio_num, output_mode)
            except Exception:
                self._dir_pin = Pin(self.dir_board_pin, output_mode)
            if self.enable_board_pin is not None:
                try:
                    self._enable_pin = Pin(self.enable_gpio_num, output_mode)
                except Exception:
                    self._enable_pin = Pin(self.enable_board_pin, output_mode)
            self._write_enable(False)
            self.ready = True
            print(
                "[Stepper:{}] step={} dir={} en={}".format(
                    self.name,
                    self.step_board_pin,
                    self.dir_board_pin,
                    self.enable_board_pin,
                )
            )
        except Exception as e:
            self._init_failed = True
            self._init_error = str(e)
            self.ready = False
            print("[Stepper:{}] init failed: {}".format(self.name, e))

    def _ensure_pwm(self):
        if self._pwm is not None or not self.ready:
            return self._pwm
        try:
            machine = __import__("machine")
            Pin = getattr(machine, "Pin")
            PWM = getattr(machine, "PWM")
            self._pwm = PWM(Pin(self.step_board_pin), freq=max(self.min_freq, 1), duty=0)
        except Exception as e:
            self._init_failed = True
            self._init_error = str(e)
            self.ready = False
            self._pwm = None
            print("[Stepper:{}] PWM init failed: {}".format(self.name, e))
        return self._pwm

    def _write_enable(self, enabled):
        if self._enable_pin is None:
            return
        if self.enable_active_low:
            self._enable_pin.value(0 if enabled else 1)
        else:
            self._enable_pin.value(1 if enabled else 0)

    def _set_direction(self, forward):
        if self._dir_pin is None:
            return
        value = 1 if forward else 0
        if self.dir_invert:
            value = 1 - value
        self._dir_pin.value(value)

    def _set_pwm(self, freq, duty):
        pwm = self._ensure_pwm()
        if pwm is None:
            return
        pwm.freq(max(1, int(freq)))
        pwm.duty(int(duty))

    def _reset_pid(self):
        self._pid_ready = False
        self._last_error = 0.0
        self._integral = 0.0
        self._derivative = 0.0
        self._last_output = 0.0

    def _compute_pid_output(self, signed_error, dt_s):
        if self._pid_ready and dt_s > 0.0:
            raw_derivative = (signed_error - self._last_error) / dt_s
            self._derivative = (
                self.derivative_alpha * self._derivative
                + (1.0 - self.derivative_alpha) * raw_derivative
            )
        else:
            self._derivative = 0.0

        integral_candidate = self._integral
        if abs(signed_error) <= self.integral_active_error:
            integral_candidate += signed_error * dt_s
            integral_candidate = _clamp(
                integral_candidate,
                -self.integral_limit,
                self.integral_limit,
            )

        unclamped_output = (
            self.pid_kp * signed_error
            + self.pid_ki * integral_candidate
            + self.pid_kd * self._derivative
        )
        clamped_output = _clamp(unclamped_output, -self.max_freq, self.max_freq)

        if unclamped_output == clamped_output:
            self._integral = integral_candidate
        else:
            same_sign = (
                (unclamped_output > self.max_freq and signed_error > 0.0)
                or (unclamped_output < -self.max_freq and signed_error < 0.0)
            )
            if not same_sign:
                self._integral = integral_candidate

        self._last_error = signed_error
        self._last_output = clamped_output
        self._pid_ready = True
        return clamped_output

    def stop(self):
        self._current_freq = 0.0
        self._reset_pid()
        if self._pwm is not None:
            try:
                self._pwm.duty(0)
            except Exception:
                pass
        if not self.hold_enabled:
            self._write_enable(False)

    def drive_error(self, error_value, allow_drive=True):
        if (not self.ready) or (not allow_drive) or (error_value is None):
            self.stop()
            self._last_update_ms = time.ticks_ms()
            return

        now = time.ticks_ms()
        dt_ms = max(1, time.ticks_diff(now, self._last_update_ms))
        self._last_update_ms = now
        dt_s = dt_ms / 1000.0

        signed_error = float(error_value) * float(self.command_sign)
        magnitude = abs(signed_error)
        if magnitude <= self.deadband:
            self.stop()
            return

        pid_output = self._compute_pid_output(signed_error, dt_s)
        if abs(pid_output) < self.min_freq:
            pid_output = self.min_freq if pid_output >= 0.0 else -self.min_freq

        self._write_enable(True)
        self._set_direction(pid_output >= 0.0)

        target_freq = abs(pid_output)
        max_delta = self.ramp_hz_per_s * dt_ms / 1000.0

        if self._current_freq <= 0.0:
            self._current_freq = float(self.min_freq)
        if target_freq > self._current_freq:
            self._current_freq = min(target_freq, self._current_freq + max_delta)
        else:
            self._current_freq = max(target_freq, self._current_freq - max_delta)

        self._set_pwm(self._current_freq, self.step_duty)

    def deinit(self):
        self.stop()
        if self._pwm is not None:
            try:
                self._pwm.deinit()
            except Exception:
                pass
            self._pwm = None
        self._write_enable(False)


class DualAxisStepperController:
    def __init__(self, axes_config):
        self.x_axis = StepperAxis(axes_config.get("x", {}))
        self.y_axis = StepperAxis(axes_config.get("y", {}))
        self.ready = self.x_axis.ready or self.y_axis.ready
        if not self.ready:
            print("[Stepper] controller inactive")

    def drive(self, error_x, error_y, allow_drive=True):
        self.x_axis.drive_error(error_x, allow_drive=allow_drive)
        self.y_axis.drive_error(error_y, allow_drive=allow_drive)

    def stop(self):
        self.x_axis.stop()
        self.y_axis.stop()

    def deinit(self):
        self.x_axis.deinit()
        self.y_axis.deinit()


def build_stepper_controller(axis_overrides=None):
    axis_overrides = axis_overrides or {}
    axes = {}
    for axis_name in DEFAULT_STEPPER_AXES:
        axes[axis_name] = _merge_axis_config(
            DEFAULT_STEPPER_AXES[axis_name],
            axis_overrides.get(axis_name),
        )
    return DualAxisStepperController(axes)
