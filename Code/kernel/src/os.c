#include <common.h>

#ifdef DEBUG_LOCAL
static sem_t empty;
static sem_t fill;
static spinlock_t print_lock;

void producer(void *arg){
  // printf("thread - %p\n", cpu_task[cpu_current()].cur_task);
  while(1){
    kmt->sem_wait(&empty);
    kmt->spin_lock(&print_lock);
    // printf("( - (%p)\n", cpu_task[cpu_current()].cur_task);
    printf("(");
    kmt->spin_unlock(&print_lock);
    kmt->sem_signal(&fill);
  }
}

void consumer(void *arg){
  while(1){
    kmt->sem_wait(&fill);
    kmt->spin_lock(&print_lock);
    printf(")");
    kmt->spin_unlock(&print_lock);
    kmt->sem_signal(&empty);
  }
}

void create_thread(){
  kmt->sem_init(&empty, "empty", 5);
  kmt->sem_init(&fill, "fill", 0);
  kmt->spin_init(&print_lock, "print lock");

  for(int i = 0 ;i < 4 ;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "producer", producer, NULL);
  for(int i = 0 ;i < 4 ;++i)
    kmt->create(pmm->alloc(sizeof(task_t)), "consumer", consumer, NULL);
}

#endif

static handler_info os_handler[MAX_HANDLER_NUM] = {0};
static int handler_num = 0;

static void os_init() {
  pmm->init();
  kmt->init();

#ifdef DEBUG_LOCAL
  create_thread();
#endif

}

static void os_run() {
#ifdef DEBUG_LOCAL
  printf("cpu (%d) running...\n", cpu_current());
  iset(true);
  while (1) printf(" idle ");
#endif
  iset(true);
  while (1) ;
}

static Context *os_trap(Event ev, Context *ctx) {
  Context *next = NULL;
  for(int i = 0 ;i < handler_num ;++i){
    handler_info *h = &os_handler[i];
    if(h->event == ev.event || h->event == EVENT_NULL){
      Context *r = h->handler(ev, ctx);
      panic_on(r && next, "returning multiple contexts");
      if (r) next = r;
    }
  }
  panic_on(!next, "returning NULL context");
  // panic_on(sane_context(next), "returning to invalid context");
  return next;
}


static inline void swap_handler(int i, int j){
  handler_info tmp = os_handler[i];
  os_handler[i] = os_handler[j];
  os_handler[j] = tmp;
}

/* only called under single processor without irq, before os->run */
static void os_on_irq(int seq, int event, handler_t handler) {
  assert(handler_num < MAX_HANDLER_NUM);
  os_handler[handler_num].seq = seq;
  os_handler[handler_num].event = event;
  os_handler[handler_num].handler = handler;
  ++ handler_num;
  for(int i = handler_num - 1 ;i >= 1 ;--i)
    if(os_handler[i].seq < os_handler[i-1].seq)
      swap_handler(i-1, i);
    else
      break;
}

MODULE_DEF(os) = {
  .init = os_init,
  .run  = os_run,
  .trap = os_trap,
  .on_irq = os_on_irq
};
