#include "vision_control_protocol.h"

#include <stdio.h>
#include <string.h>

/*
 * This file is intentionally a stub.
 * In early bring-up, the little-core Python side writes JSON into ShareFS.
 * When moved into a real K230 SDK project, replace this file with:
 * 1. a lightweight JSON parser for ShareFS polling, or
 * 2. a mailbox / IPCMSG receiver using the official inter-core API.
 */

int sharefs_json_receiver_read_latest(
    const char *path,
    vision_control_packet_t *packet
)
{
    FILE *fp;

    if (path == NULL || packet == NULL) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -2;
    }

    /*
     * TODO:
     * Parse JSON fields and map them into vision_control_packet_t.
     * For now the function only proves the file exists.
     */
    memset(packet, 0, sizeof(*packet));
    fclose(fp);
    return 0;
}
