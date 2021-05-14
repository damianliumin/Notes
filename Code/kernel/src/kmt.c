#include <common.h>


#define current (cpu_task[cpu_current()].cur_task)

cpu_info cpu_task[CPU_NUM] = {0};
task_t cpu_idle[CPU_NUM] = {0};
task_t *task_list_cur = NULL;
spinlock_t task_list_lock;

static inline void check_canary(task_t* task){
  assert(ienabled() == false);
  assert(task != &cpu_idle[cpu_current()]);
  Assert(*(uint32_t*)task->stack == CANARY, 6, "canary check fail\n");
}

static inline void set_canary(task_t* task){
  *(uint32_t*)task->stack = CANARY;
}

static int kmt_create(task_t *task, const char *name, void (*entry)(void *arg), void *arg) {
  /* not on irq */
  // basic init
  memset(task, 0, sizeof(task_t));
  task->status = SLEEP;
  task->stack = pmm->alloc(STACK_SIZE);
  set_canary(task);
  Area area = {task->stack, task->stack + STACK_SIZE};
  task->context = kcontext(area, entry, arg);
  task->name = name;
  task->sem_next = NULL;
  task->cpu = -1;
  // add to task list
  spin_lock(&task_list_lock);
  if(task_list_cur == NULL){
    task_list_cur = task;
    task->next = task->prev = task;
  } else {
    task->next = task_list_cur->next;
    task->prev = task_list_cur;
    task->next->prev = task;
    task->prev->next = task;
  }
  spin_unlock(&task_list_lock);
  return 0;
}

static void kmt_teardown(task_t *task) {
  pmm->free(task->stack);
  // remove from task list
  spin_lock(&task_list_lock);
  if(task->next == task){
    assert(task_list_cur == task);
    task_list_cur = NULL;
  } else {
    if(task_list_cur == task)
      task_list_cur = task->next;
    task->next->prev = task->prev;
    task->prev->next = task->next;
  }
  spin_unlock(&task_list_lock);
}

static task_t *get_task(){  // only get_task can change status from SLEEP to RUN
  spin_lock(&task_list_lock);
  assert(ienabled() == false);
  task_t *ret = NULL;
  task_t *rec = task_list_cur;
  if(rec)
    assert((void*)rec >= heap.start && (void*)rec <= heap.end);
  if(task_list_cur != NULL){
    do {
      if(task_list_cur->status == SLEEP && task_list_cur->cpu == -1){  // caution!
        ret = task_list_cur;
        ret->status = RUN;
        ret->cpu = cpu_current();
        task_list_cur = task_list_cur->next;
        break;
      }
      task_list_cur = task_list_cur->next;
    } while(task_list_cur != rec);
  }

  // task_t *cur = task_list_cur;
  // Log("#%d: ", cpu_current());
  // if(task_list_cur != NULL){
  //   do{
  //     Log("%p(#%d,%d)-> ", cur, cur->cpu, cur->status);
  //     cur = cur->next;
  //   } while(cur != task_list_cur);
  //   Log("\n");
  // } else 
  //   Log("\n");

  spin_unlock(&task_list_lock);
  return ret;
}

static Context* kmt_context_save(Event ev, Context *context){
  // Log("%p save context (#%d)\n", current, cpu_current());
  if(current != &cpu_idle[cpu_current()]) check_canary(current);
  spin_lock(&task_list_lock);
  current->context = context;
  assert(current->cpu == cpu_current());
  current->cpu = -1;

  if(current->status == RUN)
    current->status = SLEEP;
  if(ev.event == EVENT_ERROR)
    current->status = ERROR;
  spin_unlock(&task_list_lock);
  assert(ienabled() == false);
  // Log("return from save (#%d)\n", cpu_current());
  return NULL;
}

static Context* kmt_schedule(Event ev, Context *context){
  // Log("#%d enter schd\n", cpu_current());
  task_t *to = get_task();
  // Log("#%d got task\n", cpu_current());
  assert(ienabled() == false);
  if(to == NULL){
    to = &cpu_idle[cpu_current()];
    to->status = RUN;
    to->cpu = cpu_current();
  } else 
    check_canary(to);
  
  // Log("%p schd to %p (#%d)\n", current, to, cpu_current());

  assert(to->status == RUN);
  assert(to->context);
  assert(to->cpu == cpu_current());

  current = to;
  // Log("#%d leave schd\n", cpu_current());
  return to->context;
}

static Context* kmt_error(Event ev, Context *context){
  assert(ev.event == EVENT_ERROR);
  debug_sign(7);
  Log("Error!\n%s", ev.msg);
  Log("$rsp: %x $rip: %x\n$rsp0: %x $rflags: %x\n", context->rsp, context->rip,  context->rsp0, context->rflags);
  assert(0);
  return NULL;
}

static void kmt_init(){
  for(int i = 0 ;i < cpu_count() ;++i){
    cpu_task[i].cur_task = &cpu_idle[i];
    cpu_idle[i].name = "idle";
    cpu_idle[i].next = cpu_idle[i].prev = NULL;
    cpu_idle[i].status = RUN;
    cpu_idle[i].context = NULL;
    cpu_idle[i].stack = NULL;
    cpu_idle[i].cpu = i;
  }

  spin_init(&task_list_lock, "task list lock");
  os->on_irq(INT_MIN, EVENT_NULL, kmt_context_save);   // first call
  os->on_irq(INT_MAX, EVENT_NULL, kmt_schedule);       // last call
  os->on_irq(INT_MIN + 1, EVENT_ERROR, kmt_error);
}

MODULE_DEF(kmt) = {
  .init = kmt_init,
  .create = kmt_create,  // not on irq
  .teardown = kmt_teardown,  // not on irq
  .spin_init = spin_init,
  .spin_lock = spin_lock,  // on irq
  .spin_unlock = spin_unlock,  // on irq
  .sem_init = sem_init,
  .sem_wait = sem_wait,  // not on irq
  .sem_signal = sem_signal,  // on irq
};

