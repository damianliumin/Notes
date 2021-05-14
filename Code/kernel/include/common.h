#ifndef COMMON_H_
#define COMMON_H_
#include <kernel.h>
#include <klib.h>
#include <klib-macros.h>
 
#define CPU_NUM 8

#define MB * 1024 * 1024
#define KB * 1024

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

#define INT_MAX 2147483647
#define INT_MIN (-2147483648)

/********** DEBUG **********/
#define DEBUG_LOCAL

#ifdef DEBUG_LOCAL
extern spinlock_t print_lock;
extern task_t *task_list_cur;

void debug_sign(int);  
void debug_init();
void test_basic_kmt();
void test_spinlock();
void test_create_teardown();
void create_thread();

#define Log(...) \
  { spin_lock(&print_lock);\
  printf(__VA_ARGS__);\
  spin_unlock(&print_lock); }

#define Trace(s)\
  { \
    spin_lock(&print_lock); \
    putstr(s); \
    spin_unlock(&print_lock); \
  }

#define Assert(cond, sign, ...) \
  { \
    if(!(cond)){ \
      debug_sign(sign); \
      putstr(__VA_ARGS__); \
    } \
  }


#else
#define Log(...) {(void)0;}
#define Trace(...) {(void)0;}
#define Assert(...) {(void)0;}
#endif

/********** spinlock **********/
typedef struct spinlock {
  intptr_t locked;
  const char *name;
  int cpu;
} spinlock_t;

void spin_init(spinlock_t *lk, const char *name);
void spin_lock(spinlock_t *lk);
void spin_unlock(spinlock_t *lk);
bool holding_spinlock(spinlock_t *lk);

/********** pmm **********/
#define SAFE_PMM

#ifndef SAFE_PMM
#define HDR_SIZE 256
#define PAGE_SIZE (32 KB)
#define SLAB_SIZE (16 MB)
#define MAX_SLAB_NUM 32

#define SLOW_SLAB_SIZE  SLAB_SIZE
#define MAX_SLOW_SLAB_NUM  MAX_SLAB_NUM
#define FAST_SLAB_TYPE_NUM 12

#define FULL 0xffffffff
#define EMPTY 0x0

#define FAST 1
#define SLOW 2

typedef union page {
  struct {
    spinlock_t lock; // lock, for sequential distribute and mult free
    void* obj_start;
    union page *prev, *next;
    int cpu;
    int obj_cnt;     // allocated obj in page
    int obj_size;
    int bit_num;
    uint32_t bitmap[50];
    // 256 - (8 * 3 + 4 * 4 + 4 * 50) = 16
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

/********** kmt **********/
#define MAX_HANDLER_NUM 64
#define STACK_SIZE (16 KB)
#define CANARY 0x98765432

typedef struct handler_info{  // for os->trap
  int seq;
  int event;
  handler_t handler;
} handler_info;

typedef struct cpu_info{
  task_t *cur_task;
  int num_lock;
  bool init_ienable;
} cpu_info;

extern cpu_info cpu_task[CPU_NUM];

enum task_status { RUN = 1, SLEEP, BLOCK, ERROR};

typedef struct task{
  task_t *next, *prev;
  enum task_status status;
  Context *context;
  task_t *sem_next;
  void *stack;
  const char *name;
  int cpu;
} task_t;

/********** semaphore **********/
typedef struct semaphore{
  const char *name;
  int count;
  spinlock_t sem_lock;
  task_t wait_head;
} sem_t;

void sem_init(sem_t *sem, const char *name, int value);
void sem_wait(sem_t *sem);
void sem_signal(sem_t *sem);

#endif