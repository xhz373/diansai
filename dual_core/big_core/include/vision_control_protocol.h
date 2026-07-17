#ifndef VISION_CONTROL_PROTOCOL_H
#define VISION_CONTROL_PROTOCOL_H

#include <stdint.h>

#define VISION_CONTROL_MAGIC 0x4B323343u
#define VISION_CONTROL_VERSION 1u

typedef enum {
    VISION_MODE_IDLE = 0,
    VISION_MODE_STAND = 1,
    VISION_MODE_AIM = 2,
    VISION_MODE_CIRCLE = 3,
} vision_mode_t;

typedef enum {
    VISION_UNIT_NONE = 0,
    VISION_UNIT_PIXEL = 1,
    VISION_UNIT_CM = 2,
} vision_unit_t;

typedef enum {
    VISION_STATE_IDLE = 0,
    VISION_STATE_WAITING = 1,
    VISION_STATE_RUNNING = 2,
    VISION_STATE_STOPPED = 3,
    VISION_STATE_TRACKING = 4,
} vision_state_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t seq;
    uint32_t timestamp_ms;
    uint32_t mode;
    uint32_t unit;
    uint32_t valid;
    uint32_t control_enabled;
    uint32_t sync_ok;
    uint32_t aligned;
    uint32_t state;
    int32_t error_x_milli;
    int32_t error_y_milli;
    int32_t target_x_milli;
    int32_t target_y_milli;
} vision_control_packet_t;

#endif
