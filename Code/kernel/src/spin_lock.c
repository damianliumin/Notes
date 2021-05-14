#include <common.h>

void spin_init(spinlock_t *lk, const char *name) {
  lk->locked = 0;
  lk->name = name;
  lk->cpu = -1;
}

inline bool holding_spinlock(spinlock_t *lk){
  return lk->cpu == cpu_current();
}

void spin_lock(spinlock_t *lk) {
  bool i = ienabled();
  __sync_synchronize();
  iset(false);
  if(cpu_task[cpu_current()].num_lock == 0)
    cpu_task[cpu_current()].init_ienable = i;

  Assert(!holding_spinlock(lk), 0, 
        "AA dead lock\n");  

  while (atomic_xchg((int*)&lk->locked, 1)) ;  // fence
  __sync_synchronize();

  cpu_task[cpu_current()].num_lock ++;
  
  Assert(lk->cpu == -1, 1, 
        "lock conflict\n");

  lk->cpu = cpu_current();
}

void spin_unlock(spinlock_t *lk) {
  assert(lk->cpu == cpu_current());
  lk->cpu = -1;
  assert(ienabled() == false);
  __sync_synchronize();
  // atomic_xchg((int*)&lk->locked, 0);
  asm volatile ("movl $0, %0" : "+m"(lk->locked) : );
  cpu_task[cpu_current()].num_lock --;
  if(cpu_task[cpu_current()].num_lock == 0
  && cpu_task[cpu_current()].init_ienable){
    assert(ienabled() == false);
    iset(true);
  }
}



