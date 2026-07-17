from dual_core_ipc import MODE_AIM
from dual_core_ipc import MODE_CIRCLE
from dual_core_ipc import MODE_IDLE
from dual_core_ipc import MODE_STAND
from dual_core_ipc import UNIT_CM
from dual_core_ipc import UNIT_NONE
from dual_core_ipc import UNIT_PIXEL
from dual_core_ipc import VisionControlPublisher
from k230_common import build_stepper_controller


BACKEND_LOCAL = "local"
BACKEND_SHAREFS_IPC = "sharefs_ipc"


class LocalMotorControlBackend:
    def __init__(self, axis_overrides):
        self.motor = build_stepper_controller(axis_overrides)
        self.ready = self.motor.ready

    def update(
        self,
        error_x,
        error_y,
        valid,
        control_enabled,
        target_x=None,
        target_y=None,
        sync_ok=True,
        aligned=False,
        state_name="IDLE",
    ):
        if valid:
            self.motor.drive(
                error_x,
                error_y,
                allow_drive=control_enabled and sync_ok and (not aligned),
            )
        else:
            self.motor.stop()

    def stop(self, state_name="STOPPED"):
        self.motor.stop()

    def deinit(self):
        self.motor.deinit()


class SharefsIPCControlBackend:
    def __init__(self, mode_code, unit_code):
        self.mode_code = mode_code
        self.unit_code = unit_code
        self.publisher = VisionControlPublisher()
        self.ready = self.publisher.ready

    def update(
        self,
        error_x,
        error_y,
        valid,
        control_enabled,
        target_x=None,
        target_y=None,
        sync_ok=True,
        aligned=False,
        state_name="IDLE",
    ):
        self.publisher.publish(
            mode_code=self.mode_code,
            unit_code=self.unit_code,
            error_x=error_x,
            error_y=error_y,
            target_x=target_x,
            target_y=target_y,
            valid=valid,
            control_enabled=control_enabled,
            sync_ok=sync_ok,
            aligned=aligned,
            state_name=state_name,
        )

    def stop(self, state_name="STOPPED"):
        self.publisher.publish_stop(
            mode_code=self.mode_code,
            unit_code=self.unit_code,
            state_name=state_name,
        )

    def deinit(self):
        self.stop()


def build_control_backend(backend_name, axis_overrides=None, mode_code=MODE_IDLE, unit_code=UNIT_NONE):
    if backend_name == BACKEND_SHAREFS_IPC:
        return SharefsIPCControlBackend(mode_code=mode_code, unit_code=unit_code)
    return LocalMotorControlBackend(axis_overrides or {})


def mode_code_for_name(name):
    mapping = {
        "stand": MODE_STAND,
        "aim": MODE_AIM,
        "circle": MODE_CIRCLE,
        "idle": MODE_IDLE,
    }
    return mapping.get(str(name).lower(), MODE_IDLE)


def unit_code_for_name(name):
    mapping = {
        "pixel": UNIT_PIXEL,
        "cm": UNIT_CM,
        "none": UNIT_NONE,
    }
    return mapping.get(str(name).lower(), UNIT_NONE)
