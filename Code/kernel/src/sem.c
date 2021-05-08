#include <common.h>

#ifdef DEBUG_LOCAL
// #define SEM_LOG
#endif

#define current (cpu_task[cpu_current()].cur_task)

void sem_init(sem_t *sem, const char *name, int value) {
    sem->name = name;
    sem->count = value;
    sem->wait_head.sem_next = NULL;
    spin_init(&sem->sem_lock, name);
}

void sem_wait(sem_t *sem) {  // P
    spin_lock(&sem->sem_lock);
    sem->count --;
    int success = 1;
    if(sem->count < 0){
        success = 0;
        current->status = BLOCK;
        current->sem_next = NULL;
        task_t *cur = &sem->wait_head;
        while(cur->sem_next != NULL)
            cur = cur->sem_next;
        cur->sem_next = current;
    }

#ifdef SEM_LOG
    if(success)
        printf("pass %s count:%d->%d (%p)\n", sem->name, sem->count + 1, sem->count, current);
    else
        printf("fail %s count:%d->%d (%p)\n", sem->name, sem->count + 1, sem->count, current);
#endif

    spin_unlock(&sem->sem_lock);
    if(!success) {
        yield();
    }
}

void sem_signal(sem_t *sem) {  // V
    spin_lock(&sem->sem_lock);
    sem->count ++;
    if(sem->wait_head.sem_next){
        sem->wait_head.sem_next->status = SLEEP;
        sem->wait_head.sem_next = sem->wait_head.sem_next->sem_next;
    }
    spin_unlock(&sem->sem_lock);
}


