#ifndef COMMON_H_
#define COMMON_H_
#include <kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef TEST
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define HEAP_SIZE 128 * 1024 * 1024

struct heap{
  void *start, *end;
}heap;
#endif

#define MB * 1024 * 1024
#define KB * 1024

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

/********** spin-lock **********/
typedef struct spinlock {
  intptr_t locked;
} spinlock_t;

void spin_init(spinlock_t *lk);
void spin_lock(spinlock_t *lk);
void spin_unlock(spinlock_t *lk);

/********** pmm **********/
#define HDR_SIZE 256
#define PAGE_SIZE (32 KB)
#define SLAB_SIZE (16 MB)
#define MAX_SLAB_NUM 32

#define SLOW_SLAB_SIZE  SLAB_SIZE
#define MAX_SLOW_SLAB_NUM  MAX_SLAB_NUM
#define FAST_SLAB_TYPE_NUM 12

#define CPU_NUM 8
#define FULL 0xffffffff
#define EMPTY 0x0

#define FAST 1
#define SLOW 2

typedef union page {
  struct {
    int cpu;
    spinlock_t lock; // lock, for sequential distribute and mult free
    int obj_cnt;     // allocated obj in page
    int obj_size;
    void* obj_start;
    union page *prev, *next;
    int bit_num;
    uint32_t bitmap[50]; 
    // int free_idx;
  };  // header size <= 256B
  struct {
    uint8_t header[HDR_SIZE];
    uint8_t data[PAGE_SIZE - HDR_SIZE];
  } __attribute__((packed));
} page_t;

typedef struct kheap_slab{
  uint8_t used;
  void* start;
} kheap_slab_t;

typedef struct kheap_fast{
  spinlock_t lock;
  page_t *head[FAST_SLAB_TYPE_NUM];
} kheap_fast_t;

typedef struct kheap_slow{
  spinlock_t lock;
  uint8_t cpu;
  uint8_t valid;
  void* start;
  uint16_t map[SLOW_SLAB_SIZE / PAGE_SIZE]; // 512 * 32KB = 16MB
} kheap_slow_t;

#endif