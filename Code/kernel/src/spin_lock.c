#include <common.h>

#ifdef TEST
static int atomic_xchg(volatile int *addr,
                               int newval) {
  // swap(*addr, newval);
  int result;
  asm volatile ("lock xchg %0, %1":
    "+m"(*addr), "=a"(result) : "1"(newval) : "cc");
  return result;
}
#endif

void spin_init(spinlock_t *lk) {
  lk->locked = 0;
}

void spin_lock(spinlock_t *lk) {
  while (atomic_xchg((int*)&lk->locked, 1));
}

void spin_unlock(spinlock_t *lk) {
  atomic_xchg((int*)&lk->locked, 0);
}

