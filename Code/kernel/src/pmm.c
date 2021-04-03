#include <common.h>

#ifdef TEST
/***** TEST *****/
#include <pthread.h>

static int cpu_cnt = 0;
static pthread_t cpu[CPU_NUM];
static spinlock_t cpu_lock;

static inline int pid2cpu(pthread_t tid){
  spin_lock(&cpu_lock);
  for(int i = 0 ;i < cpu_cnt ;++i)
    if(cpu[i] == tid){
      spin_unlock(&cpu_lock);
      return i;
    }
  cpu[cpu_cnt ++] = tid;
  spin_unlock(&cpu_lock);
  return cpu_cnt - 1;
}

int cpu_current(){
  return pid2cpu(pthread_self());
}

// static int ii = 0;
#endif



static void *heap_start, *heap_end;
static void *slow_slab_start[SLOW_SLAB_NUM];

static kheap_fast_t fast_slab_manager[CPU_NUM];
static kheap_slow_t slow_slab_manager[SLOW_SLAB_NUM];

static page_t *free_page;
static spinlock_t fast_page_lock;

static inline size_t idx2size(int idx){
  assert(0 <= idx && idx < FAST_SLAB_TYPE_NUM); 
  return 1 << (idx + 3);  // 8B ~ 16KB
}

static inline int size2idx(size_t size){
  size >>= 4;
  int ret = 0;
  while(size){
    size >>= 1;
    ret ++;
  }
  return ret;
}

static inline size_t allign_size(size_t size){
  size_t ret = 1;
  while(ret < size)
    ret <<= 1;
  return ret;
}

/* slab manager */
static page_t* alloc_fast_page(size_t size, int cpu){
  spin_lock(&fast_page_lock);
  if(free_page){
    page_t *ret = free_page;
    // printf("*** get page: %p (%d) cpu %d\n", ret, (int)(ret - (page_t*)heap_start), cpu);
    free_page = free_page->next;
    spin_unlock(&fast_page_lock);
    /* init new page */
    memset((void*)ret, 0, sizeof(page_t));
    ret->cpu = cpu;
    ret->obj_size = size;
    ret->obj_cnt = 0;
    ret->next = ret->prev = NULL;
    if(size <= 256)
      ret->obj_start = (void*)ret->data;
    else
      ret->obj_start = (void*)((char*)ret + size);
    ret->bit_num = ((uintptr_t)(ret+1) - (uintptr_t)ret->obj_start) / size;
    if(ret->bit_num > 50 * 32)
      ret->bit_num = 50 * 32;
    spin_unlock(&fast_page_lock);
    return ret;
  } else {
    spin_unlock(&fast_page_lock);
    return NULL;
  }
}

static void free_fast_page(page_t *page){
  spin_lock(&fast_page_lock);
  // printf("*** return page %p (%d) cpu %d\n", page,(int)(page- (page_t*)heap_start), page->cpu);
  if(page->prev == NULL)
    fast_slab_manager[page->cpu].head[size2idx(page->obj_size)] = page->next;
  else
    page->prev->next = page->next;
  if(page->next != NULL)
    page->next->prev = page->prev;
  page->next = free_page;
  page->prev = NULL;
  if(free_page) free_page->prev = page;
  free_page = page;
  spin_unlock(&fast_page_lock);
}

/* fast path */
static void* fast_alloc(size_t size, int cpu){
  page_t* cur_page = fast_slab_manager[cpu].head[size2idx(size)];
  while(cur_page != NULL){
    if(cur_page->obj_cnt < cur_page->bit_num)
      for(int i = 0 ;i < (cur_page->bit_num + 31) / 32 ;++i){
        if(cur_page->bitmap[i] == FULL) continue;
        int t = 0;
        while(cur_page->bitmap[i] & (1 << t)) t++;
        if(i * 32 + t + 1 > cur_page->bit_num) break;
        cur_page->bitmap[i] |= (1 << t);
        cur_page->obj_cnt ++;
        #ifdef TEST
        // printf("#%d kalloc\t%ld\t%p\t%p\t%d\n", ii++, size, 
                // (char*)cur_page->obj_start + (i * 32 + t) * size, 
                // (char*)cur_page->obj_start + (i * 32 + t) * size + size, cpu);
        #endif
        return (char*)cur_page->obj_start + (i * 32 + t) * size;
      }
    else
      cur_page = cur_page->next;
  }
  #ifdef TEST
  // printf("#%d kalloc\t%ld\tfail\t%d\n", ii++, size, cpu);
  #endif
  return NULL;
}

static void fast_free(void* ptr){
  page_t *page = (page_t*)((char*)ptr - (uintptr_t)ptr % PAGE_SIZE);
  int cpu = page->cpu;
  spin_lock(&fast_slab_manager[cpu].lock);
  int idx = (int)((char*)ptr - (char*)page->obj_start) / page->obj_size;
  int i = idx / 32;
  if(page->bitmap[i] & (1 << (idx % 32))){
    page->obj_cnt --;
    page->bitmap[i] &= ~(1 << (idx % 32));
    if(page->obj_cnt == 0)
      free_fast_page(page);
  }
  #ifdef TEST
  // printf("#%d kfree\t%p\t%d\n", ii++, ptr, cpu);
  #endif
  spin_unlock(&fast_slab_manager[cpu].lock);
}

/* slow path */
static void* slow_alloc(size_t size, int id){
  spin_lock(&slow_slab_manager[id].lock);
  char* start = slow_slab_start[id];
  int map_cnt = size / (32 KB);
  for(int i = 0 ;i < SLOW_SLAB_SIZE / PAGE_SIZE ;i += map_cnt){
    int valid = 1;
    for(int j = i ;j < i + map_cnt ; ++j)
      if(slow_slab_manager[id].map[j]){
        valid = 0;
        break;
      }
    if(!valid) continue;
    for(int j = i ;j < i + map_cnt ; ++j)
      slow_slab_manager[id].map[j] = map_cnt - (j - i);
    spin_unlock(&slow_slab_manager[id].lock); 
    return start + i * (32 KB);
  }
  spin_unlock(&slow_slab_manager[id].lock);
  return NULL;
}

static void slow_free(void* ptr, int id){
  spin_lock(&slow_slab_manager[id].lock);
  int start_idx = (int)((char*)ptr - (char*)slow_slab_start[id]) / PAGE_SIZE;
  int map_cnt = slow_slab_manager[id].map[start_idx];
  for(int i = start_idx ;i < start_idx + map_cnt; ++i)
    slow_slab_manager[id].map[i] = 0;
  spin_unlock(&slow_slab_manager[id].lock);
}

/* kalloc, kfree, pmm_init */
static void *kalloc(size_t size) {
  if (size > 16 MB)  // reject memory >= 16 MB
    return NULL;
  size = allign_size(size);
  size = MAX(size, 8);
  if (size <= 16 KB){  // fast path
    int cpu = cpu_current();
    // printf("cpu: %d\n", cpu);
    spin_lock(&fast_slab_manager[cpu].lock);
    void* ret = fast_alloc(size, cpu);
    if(!ret) {
      page_t* new_page = alloc_fast_page(size, cpu);
      // printf("***[ch] get page %p (%d) cpu %d\n", new_page,(int)(new_page- (page_t*)heap_start), new_page->cpu);
      if(new_page == NULL) return NULL;
      page_t* head = fast_slab_manager[cpu].head[size2idx(size)];
      new_page->next = head;
      if(head) head->prev = new_page;
      fast_slab_manager[cpu].head[size2idx(size)] = new_page;
      ret = fast_alloc(size, cpu);
    }
    spin_unlock(&fast_slab_manager[cpu].lock);
    return ret;
  } else{  // slow path
    int id = rand() % 2;
    for(int i = 0 ;i < SLOW_SLAB_NUM ;++i){
      void *ret = slow_alloc(size, (id + i) % SLOW_SLAB_NUM);
      if(ret) return ret;
    }
    return NULL;
  }  
}

static void kfree(void *ptr) {
  if((char*)ptr < (char*)heap_start || (char*)ptr >= (char*)heap_end) return;
  if((char*)ptr < (char*)slow_slab_start[0]){
    fast_free(ptr);
  } else {
    for(int i = 0 ;i < SLOW_SLAB_NUM ;++i)
      if((char*)ptr - (char*)slow_slab_start[i] < SLOW_SLAB_SIZE){
        slow_free(ptr, i);
        return;
      }
    assert(0);
  }
}

static void pmm_init() {
  #ifndef TEST
  uintptr_t pmsize = ((uintptr_t)heap.end - (uintptr_t)heap.start);
  printf("Got %d MiB heap: [%p, %p)\n", pmsize >> 20, heap.start, heap.end);
  #else
  char *ptr  = malloc(HEAP_SIZE);
  heap.start = ptr;
  heap.end   = ptr + HEAP_SIZE;
  printf("Got %d MiB heap: [%p, %p)\n", HEAP_SIZE >> 20, heap.start, heap.end);
  spin_init(&cpu_lock);
  #endif
  heap_start = (void*)((uintptr_t)heap.start + PAGE_SIZE - (uintptr_t)heap.start % (PAGE_SIZE));
  heap_end = (void*)((uintptr_t)heap.end - (uintptr_t)heap.end % (16 MB));
  printf("heap_start: %p, heap_end: %p\n", heap_start, heap_end);
  /* init slow page */
  for(int i = 0 ;i < SLOW_SLAB_NUM ;++i){
    slow_slab_start[i] = (void*)((uintptr_t)heap_end - (SLOW_SLAB_NUM - i) * SLOW_SLAB_SIZE);
    memset(slow_slab_start[i], SLOW_SLAB_SIZE, 0);
    spin_init(&slow_slab_manager[i].lock);
  }
  /* init fast page */
  spin_init(&fast_page_lock);
  free_page = (page_t*)heap_start;
  page_t* tmp = free_page;
  while(tmp < (page_t*)slow_slab_start[0]){
    spin_init(&tmp->lock);
    if(tmp > free_page) tmp->prev = tmp - 1;
    if(tmp + 1 < (page_t*)slow_slab_start[0]) tmp->next = tmp + 1;
    tmp = tmp + 1;
  }
}

MODULE_DEF(pmm) = {
  .init  = pmm_init,
  .alloc = kalloc,
  .free  = kfree,
};

