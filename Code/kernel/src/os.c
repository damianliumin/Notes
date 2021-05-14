#include <common.h>

static handler_info os_handler[MAX_HANDLER_NUM] = {0};
static int handler_num = 0;
static spinlock_t os_trap_lock;

static void os_init() {
  pmm->init();
  kmt->init();
  spin_init(&os_trap_lock, "os trap lock");


#ifdef DEBUG_LOCAL
  debug_init();
  // test_spinlock();
  // test_basic_kmt();
  // test_create_teardown();
  create_thread();
#endif
}

static void os_run() {
  iset(true);
  while (1) ;
}

static Context *os_trap(Event ev, Context *ctx) {
  if(!holding_spinlock(&os_trap_lock)){  // normal trap
    Trace("t");
    spin_lock(&os_trap_lock);
    Assert(ienabled() == false, 3, "i opened in trap\n");
    volatile Context *next = NULL;
    int i;
    for(i = 0 ;i < handler_num ;++i){
      handler_info h = os_handler[i];
      if(h.event == ev.event || h.event == EVENT_NULL){
        Context *r = h.handler(ev, ctx);
        Assert(!(r && next), 4, "returning multiple contexts");
        if (r) next = r;
      }
    }
    Assert(next, 5, "returning NULL context");
    spin_unlock(&os_trap_lock);
    return next;
  } else {
    Trace(ev.msg);
    Assert(0, 6, "kernel error\n");

    halt(0);
  }
}

static inline void swap_handler(int i, int j){
  handler_info tmp = os_handler[i];
  os_handler[i] = os_handler[j];
  os_handler[j] = tmp;
}

/* only called under single processor without irq, before os->run */
static void os_on_irq(int seq, int event, handler_t handler) {
  assert(handler_num < MAX_HANDLER_NUM);
  assert(ienabled() == false);
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
