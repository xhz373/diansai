import sys
from machine import FPIOA, Pin

from common_hw import map_gpio, pin_pull_up

FLAG = 2
FLAG_SEQUENCE = (0, 1, 2, 3, 4, 5, 6)
IDLE_FLAG = 1
FLAG_MAP = {
    0: ("calibrate", "calibrate"),
    2: ("stand_aiming", "stand_aiming"),
    3: ("stand_aiming", "stand_aiming"),
    4: ("move_aiming", "move_aiming"),
    5: ("move_aiming", "move_aiming"),
    6: ("circle_mode", "circle_mode"),
}

MODE_BUTTON_CONFIGS = (
    ("MODE_BTN", 20, 20),
)
MODE_BUTTON_DEBOUNCE_MS = 35
MODE_BUTTON_BOOT_DETECT_MS = 120
MODE_BUTTON_SELECT_TIMEOUT_MS = 900
MODE_BUTTON_POLL_MS = 20


def _next_flag_value(current_flag):
    try:
        index = FLAG_SEQUENCE.index(int(current_flag))
    except Exception:
        index = -1
    return FLAG_SEQUENCE[(index + 1) % len(FLAG_SEQUENCE)]


def _select_flag_from_gpio(default_flag):
    try:
        machine = __import__("machine")
        time_mod = __import__("time")
        FPIOA = getattr(machine, "FPIOA")
        Pin = getattr(machine, "Pin")
    except Exception:
        return default_flag

    try:
        fpioa = FPIOA()
        pull_up = pin_pull_up()
        buttons = []
        for key_name, board_pin, gpio_num in MODE_BUTTON_CONFIGS:
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
                buttons.append((key_name, pin))
            except Exception:
                pass

        if not buttons:
            return default_flag

        pressed_at_boot = False
        detect_start_ms = time_mod.ticks_ms()
        while time_mod.ticks_diff(time_mod.ticks_ms(), detect_start_ms) < MODE_BUTTON_BOOT_DETECT_MS:
            for _, pin in buttons:
                try:
                    if pin.value() == 0:
                        pressed_at_boot = True
                        break
                except Exception:
                    pass
            if pressed_at_boot:
                break
            if hasattr(time_mod, "sleep_ms"):
                time_mod.sleep_ms(MODE_BUTTON_POLL_MS)

        if not pressed_at_boot:
            return default_flag

        selected_flag = _next_flag_value(default_flag)
        print("[Launcher] MODE_BTN select start, flag {}".format(selected_flag))

        stable_value = 0
        last_raw_value = 0
        last_change_ms = time_mod.ticks_ms()
        select_deadline = time_mod.ticks_add(last_change_ms, MODE_BUTTON_SELECT_TIMEOUT_MS)
        mode_pin = buttons[0][1]

        while time_mod.ticks_diff(select_deadline, time_mod.ticks_ms()) > 0:
            try:
                raw_value = mode_pin.value()
            except Exception:
                raw_value = stable_value

            now = time_mod.ticks_ms()
            if raw_value != last_raw_value:
                last_raw_value = raw_value
                last_change_ms = now
            elif time_mod.ticks_diff(now, last_change_ms) >= MODE_BUTTON_DEBOUNCE_MS:
                if raw_value != stable_value:
                    stable_value = raw_value
                    if stable_value == 0:
                        selected_flag = _next_flag_value(selected_flag)
                        print("[Launcher] MODE_BTN -> flag {}".format(selected_flag))
                    select_deadline = time_mod.ticks_add(now, MODE_BUTTON_SELECT_TIMEOUT_MS)

            if hasattr(time_mod, "sleep_ms"):
                time_mod.sleep_ms(MODE_BUTTON_POLL_MS)

        return selected_flag
    except Exception:
        return default_flag


def _resolve_flag():
    try:
        flag = int(FLAG)
    except Exception:
        raise ValueError("FLAG must be 0, 1, 2, 3, 4, 5, or 6, got: " + str(FLAG))

    flag = _select_flag_from_gpio(flag)
    if flag == IDLE_FLAG:
        return flag, "idle", None
    if flag not in FLAG_MAP:
        raise ValueError("unsupported FLAG: " + str(flag))
    return flag, FLAG_MAP[flag][0], FLAG_MAP[flag][1]


def _load_mode_module(module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    return __import__(module_name)


def run():
    flag, mode, module_name = _resolve_flag()
    print("=" * 50)
    print("K230 competition launcher")
    print("flag:", flag)
    print("boot mode:", mode)
    print("source:", module_name)
    print("=" * 50)

    if flag == IDLE_FLAG:
        print("[Launcher] flag 1 selected, launcher idle")
        return

    module = _load_mode_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(module_name + " has no main()")
    module.main()


if __name__ == "__main__":
    run()
