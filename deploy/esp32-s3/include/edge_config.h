#pragma once

// ESP32-S3-WROOM-1 N8R8 (measured on target board, Jul 2026)
// Chip: ESP32-S3 v0.2 | Flash: 8MB (GD) | Internal SRAM: 512KB | External PSRAM: 8MB Octal
#define EDGE_BOARD_NAME "ESP32-S3-WROOM-1-N8R8"
#define EDGE_CPU_MHZ 240
#define EDGE_FLASH_MB 8
#define EDGE_PSRAM_MB 8
#define EDGE_SRAM_KB 512

#ifndef EDGE_MAX_SEQ_LEN
#define EDGE_MAX_SEQ_LEN 128
#endif

#ifndef EDGE_MAX_TAGS
#define EDGE_MAX_TAGS 32
#endif

#define EDGE_BENCHMARK_WARMUP 3
#define EDGE_BENCHMARK_REPEAT 10

#ifndef EDGE_MAX_WORD_LEN
#define EDGE_MAX_WORD_LEN 20
#endif

#ifndef EDGE_HAS_ESPDL_MODEL
#define EDGE_HAS_ESPDL_MODEL 0
#endif
