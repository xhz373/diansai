import json
import os


CALIB_FILE = "/sdcard/app/aiming_calib.json"


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
