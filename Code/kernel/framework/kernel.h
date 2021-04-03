#include <am.h>

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

#define MODULE(mod) \
  typedef struct mod_##mod##_t mod_##mod##_t; \
  extern mod_##mod##_t *mod; \
  struct mod_##mod##_t

#define MODULE_DEF(mod) \
  extern mod_##mod##_t __##mod##_obj; \
  mod_##mod##_t *mod = &__##mod##_obj; \
  mod_##mod##_t __##mod##_obj

#define MB * 1024 * 1024
#define KB * 1024

#define MIN(a, b) (a < b) ? a : b
#define MAX(a, b) (a > b) ? a : b

MODULE(os) {
  void (*init)();
  void (*run)();
};

MODULE(pmm) {
  void  (*init)();
  void *(*alloc)(size_t size);
  void  (*free)(void *ptr);
};

/********** SPIN-LOCK **********/
typedef struct spinlock {
  intptr_t locked;
} spinlock_t;

void spin_init(spinlock_t *lk);
void spin_lock(spinlock_t *lk);
void spin_unlock(spinlock_t *lk);

/********** pmm **********/
#define HDR_SIZE 256
#define PAGE_SIZE (32 KB)
#define SLOW_SLAB_NUM 2
#define SLOW_SLAB_SIZE (16 MB)
#define FAST_SLAB_TYPE_NUM 12
#define CPU_NUM 8
#define FULL 0xffffffff
#define EMPTY 0x0

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
  };  // header size <= 256B
  struct {
    uint8_t header[HDR_SIZE];
    uint8_t data[PAGE_SIZE - HDR_SIZE];
  } __attribute__((packed));
} page_t;

typedef struct kheap_fast{
  spinlock_t lock;
  page_t *head[FAST_SLAB_TYPE_NUM];
} kheap_fast_t;

typedef struct kheap_slow{
  spinlock_t lock;
  uint16_t map[SLOW_SLAB_SIZE / PAGE_SIZE]; // 512 * 32KB = 16MB
} kheap_slow_t;


