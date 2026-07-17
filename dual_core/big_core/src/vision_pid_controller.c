#include "vision_pid_controller.h"

static float clamp_float(float value, float low, float high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static void reset_axis(pid_axis_state_t *axis)
{
    axis->integral = 0.0f;
    axis->derivative = 0.0f;
    axis->last_error = 0.0f;
    axis->initialized = 0u;
}

static float update_axis(pid_axis_state_t *axis, float error, float dt_s)
{
    float raw_derivative = 0.0f;
    float integral_candidate;
    float output;

    if (axis->initialized && dt_s > 0.0f) {
        raw_derivative = (error - axis->last_error) / dt_s;
        axis->derivative = axis->derivative_alpha * axis->derivative
                         + (1.0f - axis->derivative_alpha) * raw_derivative;
    } else {
        axis->derivative = 0.0f;
    }

    integral_candidate = axis->integral + error * dt_s;
    integral_candidate = clamp_float(
        integral_candidate,
        -axis->integral_limit,
        axis->integral_limit
    );

    output = axis->kp * error
           + axis->ki * integral_candidate
           + axis->kd * axis->derivative;

    axis->integral = integral_candidate;
    axis->last_error = error;
    axis->initialized = 1u;
    return output;
}

void vision_pid_controller_init(vision_pid_controller_t *controller)
{
    controller->x_axis.kp = 180.0f;
    controller->x_axis.ki = 10.0f;
    controller->x_axis.kd = 2.5f;
    controller->x_axis.integral_limit = 6.0f;
    controller->x_axis.derivative_alpha = 0.25f;

    controller->y_axis.kp = 180.0f;
    controller->y_axis.ki = 10.0f;
    controller->y_axis.kd = 2.5f;
    controller->y_axis.integral_limit = 6.0f;
    controller->y_axis.derivative_alpha = 0.25f;

    controller->min_freq_hz = 120.0f;
    controller->max_freq_hz = 1800.0f;
    controller->ramp_hz_per_s = 3200.0f;
    controller->deadband = 0.25f;
    vision_pid_controller_reset(controller);
}

void vision_pid_controller_reset(vision_pid_controller_t *controller)
{
    reset_axis(&controller->x_axis);
    reset_axis(&controller->y_axis);
}

void vision_pid_controller_update(
    vision_pid_controller_t *controller,
    const vision_control_packet_t *packet,
    float dt_s,
    vision_motor_command_t *command
)
{
    float error_x;
    float error_y;
    float output_x;
    float output_y;

    if (!packet->valid || !packet->control_enabled || !packet->sync_ok) {
        vision_pid_controller_reset(controller);
        command->x_freq_hz = 0.0f;
        command->y_freq_hz = 0.0f;
        command->x_dir_forward = 1u;
        command->y_dir_forward = 1u;
        command->enabled = 0u;
        return;
    }

    error_x = packet->error_x_milli / 1000.0f;
    error_y = packet->error_y_milli / 1000.0f;
    if (error_x < 0.0f) {
        error_x = -error_x;
    }
    if (error_y < 0.0f) {
        error_y = -error_y;
    }

    output_x = update_axis(&controller->x_axis, packet->error_x_milli / 1000.0f, dt_s);
    output_y = update_axis(&controller->y_axis, packet->error_y_milli / 1000.0f, dt_s);

    if (output_x >= 0.0f) {
        command->x_dir_forward = 1u;
    } else {
        command->x_dir_forward = 0u;
        output_x = -output_x;
    }

    if (output_y >= 0.0f) {
        command->y_dir_forward = 1u;
    } else {
        command->y_dir_forward = 0u;
        output_y = -output_y;
    }

    if (error_x <= controller->deadband) {
        output_x = 0.0f;
    } else if (output_x > 0.0f && output_x < controller->min_freq_hz) {
        output_x = controller->min_freq_hz;
    }

    if (error_y <= controller->deadband) {
        output_y = 0.0f;
    } else if (output_y > 0.0f && output_y < controller->min_freq_hz) {
        output_y = controller->min_freq_hz;
    }

    command->x_freq_hz = clamp_float(output_x, 0.0f, controller->max_freq_hz);
    command->y_freq_hz = clamp_float(output_y, 0.0f, controller->max_freq_hz);
    command->enabled = (command->x_freq_hz > 0.0f || command->y_freq_hz > 0.0f) ? 1u : 0u;
}
