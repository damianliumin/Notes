#include <common.h>

#ifdef DEBUG_LOCAL
static sem_t empty;
static sem_t fill;
spinlock_t print_lock;

volatile void debug_sign(int id) {(void)0;}

/* producer consumer */

void producer(void *arg){
  while(1){
    kmt->sem_wait(&empty);
    // Log("(-(%p, %d)\n", cpu_task[cpu_current()].cur_task, cpu_current());
    // Log("(");
    kmt->sem_signal(&fill);
  }
}

void consumer(void *arg){
  while(1){
    kmt->sem_wait(&fill);
    // Log(")-(%p, %d)\n", cpu_task[cpu_current()].cur_task, cpu_current());
    // Log(")");
    kmt->sem_signal(&empty);
  }
}

void create_thread(){
  kmt->sem_init(&empty, "empty", 5);
  kmt->sem_init(&fill, "fill", 0);

  for(int i = 0 ;i < 4 ;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "producer", producer, NULL);
  for(int i = 0 ;i < 5 ;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "consumer", consumer, NULL);
}

/* test for basic kmt */

void basic_kmt_thread(void*arg){
  printf("thread %p\n", cpu_task[cpu_current()].cur_task);
  while (1) ;
}

void test_basic_kmt(){
  for(int i = 1 ;i <= 4;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "basic kmt", basic_kmt_thread, NULL);
}

/* test for spinlock */
static spinlock_t lk1, lk2, lk3;

void spin_thread(void* arg){
  while(1){
    assert(ienabled());
    spin_lock(&lk1);
    assert(!ienabled());
    spin_lock(&lk2);
    assert(cpu_task[cpu_current()].num_lock == 2);
    assert(!ienabled());
    // printf("cpu: %d task: %p\n", cpu_current(), cpu_task[cpu_current()].cur_task);
    spin_unlock(&lk2);
    assert(cpu_task[cpu_current()].num_lock == 1);
    assert(!ienabled());
    spin_unlock(&lk1);
    /* Warning: cpu_task[cpu_current()]  compute cpu_current() and irq
     then this thread might compute cpu_task[...] with original cpu_current() */
    assert(ienabled());
    // putstr("one cycle\n")
  }
}

void test_spinlock(){
  Log("testing spinlock...\n");
  spin_init(&lk1, "lk1");
  spin_init(&lk2, "lk2");
  spin_init(&lk3, "lk3");
  for(int i = 1 ;i <= 10 ;++i){
    kmt->create(pmm->alloc(sizeof(task_t)), "spin", spin_thread, NULL);
  }
}

/* test for create and teardown */
extern spinlock_t task_list_lock;

static void idle_thread(){
  while(1);
}

static void kmanage(void *arg){
  while(1){
    task_t *t1 = pmm->alloc(sizeof(task_t));
    kmt->create(t1, "idle", idle_thread, NULL);
    spin_lock(&task_list_lock);
    Log("%p add task %p\n", cpu_task[cpu_current()].cur_task, t1);
    task_t *cur = task_list_cur;
    do{
      Log("%p(%d)-> ", cur, cpu_current());
      cur = cur->next;
    } while(cur != task_list_cur);
    Log("\n");
    spin_unlock(&task_list_lock);
    kmt->teardown(t1);
    // pmm->free(t1);
  }
}

void test_create_teardown(){
  for(intptr_t i = 0 ;i < 4 ;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "kmanage", kmanage, NULL);
}


void debug_init(){
  spin_init(&print_lock, "print lock");
}

#endif