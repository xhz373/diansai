import json
import os
import time


IPC_MAGIC = "K230CTRL"
IPC_VERSION = 1

MODE_IDLE = 0
MODE_STAND = 1
MODE_AIM = 2
MODE_CIRCLE = 3

UNIT_NONE = 0
UNIT_PIXEL = 1
UNIT_CM = 2

STATE_IDLE = 0
STATE_WAITING = 1
STATE_RUNNING = 2
STATE_STOPPED = 3
STATE_TRACKING = 4

DEFAULT_SHAREFS_ROOT = "/sharefs/k230_dual_core"
DEFAULT_PACKET_PATH = DEFAULT_SHAREFS_ROOT + "/vision_control_latest.json"
DEFAULT_SEQ_PATH = DEFAULT_SHAREFS_ROOT + "/vision_control_seq.txt"


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


def to_milli(value):
    if value is None:
        return 0
    return int(round(float(value) * 1000.0))


def _state_to_code(state_name):
    mapping = {
        "IDLE": STATE_IDLE,
        "WAITING": STATE_WAITING,
        "RUNNING": STATE_RUNNING,
        "STOPPED": STATE_STOPPED,
        "TRACKING": STATE_TRACKING,
    }
    return mapping.get(str(state_name).upper(), STATE_IDLE)


def build_control_packet(
    seq,
    mode_code,
    unit_code,
    error_x=None,
    error_y=None,
    target_x=None,
    target_y=None,
    valid=False,
    control_enabled=False,
    sync_ok=False,
    aligned=False,
    state_name="IDLE",
    timestamp_ms=None,
    source="vision",
):
    if timestamp_ms is None:
        timestamp_ms = time.ticks_ms()
    return {
        "magic": IPC_MAGIC,
        "version": IPC_VERSION,
        "seq": int(seq),
        "timestamp_ms": int(timestamp_ms),
        "source": str(source),
        "mode": int(mode_code),
        "unit": int(unit_code),
        "valid": 1 if valid else 0,
        "control_enabled": 1 if control_enabled else 0,
        "sync_ok": 1 if sync_ok else 0,
        "aligned": 1 if aligned else 0,
        "state": int(_state_to_code(state_name)),
        "error_x_milli": to_milli(error_x),
        "error_y_milli": to_milli(error_y),
        "target_x_milli": to_milli(target_x),
        "target_y_milli": to_milli(target_y),
    }


class VisionControlPublisher:
    def __init__(
        self,
        packet_path=DEFAULT_PACKET_PATH,
        seq_path=DEFAULT_SEQ_PATH,
        source="little_core_python",
    ):
        self.packet_path = packet_path
        self.seq_path = seq_path
        self.source = source
        self.seq = 0
        self.ready = False
        self._init_fs()

    def _init_fs(self):
        try:
            _ensure_parent_dir(self.packet_path)
            _ensure_parent_dir(self.seq_path)
            self.seq = self._load_seq()
            self.ready = True
            print("[IPC] sharefs ready:", self.packet_path)
        except Exception as e:
            self.ready = False
            print("[IPC] sharefs init failed:", e)

    def _load_seq(self):
        try:
            with open(self.seq_path, "r") as f:
                return int(f.read().strip() or "0")
        except Exception:
            return 0

    def _save_seq(self):
        tmp_path = self.seq_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(str(int(self.seq)))
        try:
            os.remove(self.seq_path)
        except Exception:
            pass
        os.rename(tmp_path, self.seq_path)

    def _write_packet(self, packet):
        tmp_path = self.packet_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(packet, f)
        try:
            os.remove(self.packet_path)
        except Exception:
            pass
        os.rename(tmp_path, self.packet_path)

    def publish(
        self,
        mode_code,
        unit_code,
        error_x=None,
        error_y=None,
        target_x=None,
        target_y=None,
        valid=False,
        control_enabled=False,
        sync_ok=False,
        aligned=False,
        state_name="IDLE",
    ):
        if not self.ready:
            return False
        try:
            self.seq += 1
            packet = build_control_packet(
                seq=self.seq,
                mode_code=mode_code,
                unit_code=unit_code,
                error_x=error_x,
                error_y=error_y,
                target_x=target_x,
                target_y=target_y,
                valid=valid,
                control_enabled=control_enabled,
                sync_ok=sync_ok,
                aligned=aligned,
                state_name=state_name,
                source=self.source,
            )
            self._write_packet(packet)
            self._save_seq()
            return True
        except Exception as e:
            print("[IPC] publish failed:", e)
            return False

    def publish_stop(self, mode_code, unit_code, state_name="STOPPED"):
        return self.publish(
            mode_code=mode_code,
            unit_code=unit_code,
            error_x=0.0,
            error_y=0.0,
            target_x=0.0,
            target_y=0.0,
            valid=False,
            control_enabled=False,
            sync_ok=False,
            aligned=False,
            state_name=state_name,
        )
