# K230 Dual-Core Skeleton

This directory contains a migration skeleton for splitting the current single-process
vision-and-control Python project into:

- `little_core/`: vision side, produces control error packets
- `big_core/`: real-time side, consumes packets and drives PID/PWM

Current status:

- The little-core Python side can already switch between local control and ShareFS JSON IPC.
- The big-core side is a C skeleton only. It is intended to be moved into a real K230 SDK
  RT-Smart application and to replace the ShareFS poller with mailbox/IPCMSG later.

Bring-up note:

- `ShareFS JSON` is convenient for early integration, but it is not the final real-time
  transport for competition use.
