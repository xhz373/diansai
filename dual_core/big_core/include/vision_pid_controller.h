#ifndef VISION_PID_CONTROLLER_H
#define VISION_PID_CONTROLLER_H

#include "vision_control_protocol.h"

typedef struct {
    float kp;
    float ki;
    float kd;
    float integral;
    float integral_limit;
    float derivative;
    float derivative_alpha;
    float last_error;
    uint8_t initialized;
} pid_axis_state_t;

typedef struct {
    pid_axis_state_t x_axis;
    pid_axis_state_t y_axis;
    float min_freq_hz;
    float max_freq_hz;
    float ramp_hz_per_s;
    float deadband;
} vision_pid_controller_t;

typedef struct {
    float x_freq_hz;
    float y_freq_hz;
    uint8_t x_dir_forward;
    uint8_t y_dir_forward;
    uint8_t enabled;
} vision_motor_command_t;

void vision_pid_controller_init(vision_pid_controller_t *controller);
void vision_pid_controller_reset(vision_pid_controller_t *controller);
void vision_pid_controller_update(
    vision_pid_controller_t *controller,
    const vision_control_packet_t *packet,
    float dt_s,
    vision_motor_command_t *command
);

#endif
