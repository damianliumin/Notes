#include <common.h>

void spin_init(spinlock_t *lk, const char *name) {
  lk->locked = 0;
  lk->name = name;
}

void spin_lock(spinlock_t *lk) {
  if(cpu_task[cpu_current()].num_lock == 0)
    cpu_task[cpu_current()].init_ienable = ienabled();
  iset(false);
  while (atomic_xchg((int*)&lk->locked, 1));
  cpu_task[cpu_current()].num_lock ++;
}

void spin_unlock(spinlock_t *lk) {
  atomic_xchg((int*)&lk->locked, 0);
  cpu_task[cpu_current()].num_lock --;
  if(cpu_task[cpu_current()].num_lock == 0
  && cpu_task[cpu_current()].init_ienable)
    iset(true);
}

