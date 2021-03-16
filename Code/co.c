#include "co.h"
#include <stdlib.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdio.h>
#include <assert.h>

#define STACK_SIZE 64 * 1024
#define MAX_CO_NUM 132

/* struct co */
enum co_status {
  CO_NEW = 1, // new and not executed yet
  CO_RUNNING, // executed
  CO_WAITING, // co_wait
  CO_DEAD,    // finished, resource not released yet
};

struct co {
  const char *name;
  void (*func)(void *); // entry and arg
  void *arg;

  enum co_status status;  // status of coroutine
  struct co *    waiter;  // whether waited by others
  jmp_buf        context; // reg (setjmp.h)
  uint8_t        stack[STACK_SIZE]; // stack
};

/* coroutine manager */
static struct co main_co = {"main", NULL, NULL, CO_RUNNING, NULL};
static struct co* avail[MAX_CO_NUM] = {&main_co};
static struct co* current;
static int coNum = 0;

static int addCo(struct co* co){
  avail[++coNum] = co;
  return 0;
}

static int deleteCo(struct co* co){
  for(int i = 1 ;i <= coNum ;++i)
    if(avail[i] == co){
      for(int j = i ;j < coNum ;++j){
        avail[j] = avail[j + 1];
      }
      --coNum;
      return 0;
    }
  return -1;
}

struct co* randomPick(){
  int idx = rand() % (coNum + 1);
  // printf("idx: %d ", idx);
  assert(avail[idx]);
  return avail[idx];
}

/* stack switch */
static inline void stack_switch_call(void *sp, void *entry, uintptr_t arg) {
  asm volatile (
#if __x86_64__
    "movq %0, %%rsp; movq %2, %%rdi; jmp *%1"
      : : "b"((uintptr_t)sp),     "d"(entry), "a"(arg)
#else
    "movl %0, %%esp; movl %2, 4(%0); jmp *%1"
      : : "b"((uintptr_t)sp - 8), "d"(entry), "a"(arg)
#endif
  );
}

static void co_ret(){
#if __x86_64__
  asm volatile(
    "subq $8, %%rsp;" : :
  );
#endif
  current->status = CO_DEAD;
  co_yield();
}

struct co *co_start(const char *name, void (*func)(void *), void *arg) {
  struct co *co = (struct co*)malloc(sizeof(struct co));
  *co = (struct co){name, func, arg, CO_NEW, NULL};
  addCo(co);
  return co;
}

void co_wait(struct co *co) {
  if(!current) current = &main_co;
  current->status = CO_WAITING;
  co->waiter = current;
  while(co->status != CO_DEAD){
    // printf("waiting...\n");
    co_yield();
  }
  current->status = CO_RUNNING;
  assert(!deleteCo(co));
  free(co);
}


void co_yield() {
  if(!current) current = &main_co;
  assert(current);
  int val = setjmp(current->context);
  if(val == 0){
    struct co* co = randomPick();
    while(co->status == CO_DEAD && co->waiter == NULL) co = randomPick();
    current = co;
    // printf("stautus: %d %p %p %p %p\n", co->status, co, avail[0], avail[1], avail[2]);
    switch(co->status){
      case CO_DEAD:
        current = co->waiter;
        longjmp(current->context, 1);
        break;
      case CO_NEW:
        co->status = CO_RUNNING;
      #if __x86_64__
        *(uintptr_t*)(co->stack + STACK_SIZE - sizeof(uintptr_t)) = (uintptr_t)co_ret;
        stack_switch_call(&co->stack[STACK_SIZE - sizeof(uintptr_t)], co->func, (uintptr_t)co->arg);
      #else
        *(uintptr_t*)(co->stack + STACK_SIZE - 2 * sizeof(uintptr_t)) = (uintptr_t)co_ret;  
        stack_switch_call(&co->stack[STACK_SIZE], co->func, (uintptr_t)co->arg);
      #endif
        break;
      case CO_RUNNING: case CO_WAITING:
        longjmp(co->context, 1);
        break;
      default:
        assert(0);
    }
  }
  // printf("back!\n");
}
