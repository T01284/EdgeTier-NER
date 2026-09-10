Linear-chain CRF Viterbi decoder shared by Pi 3B (Python) and ESP32-S3 (C).

- `viterbi.h` / `viterbi.c` — reference C implementation
- ESP32 copy: `deploy/esp32-s3/lib/edge_crf/`（修改时请同步）
- Pi Python port: `deploy/pi3b/edgefs_pi/crf_decode.py`
