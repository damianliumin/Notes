#include <common.h>

#ifdef TEST
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

#endif

static void *heap_start, *heap_end;
int SLAB_NUM = 0;
kheap_slab_t slab_manager[MAX_SLAB_NUM];
static spinlock_t slab_lock;

int SLOW_SLAB_NUM = 0;
kheap_slow_t slow_slab_manager[MAX_SLOW_SLAB_NUM];
static spinlock_t slow_slab_lock;

static kheap_fast_t fast_slab_manager[CPU_NUM];
static page_t *free_page = NULL;
static spinlock_t fast_page_lock;

static inline int size2idx(size_t size){
  size >>= 4;
  int ret = 0;
  while(size){
    size >>= 1;
    ret ++;
  }
  return ret;
}

static inline int first_zero_bit(uint32_t data){
  data = ~data;
  assert(data);
	int pos=0;
	if ((data & 0xFFFF) == 0)	
		data >>= 16, pos += 16;
	if ((data & 0xFF) == 0)
		data >>= 8, pos += 8;
	if ((data & 0xF) == 0)
		data >>= 4, pos += 4;
	if ((data & 0x3) == 0)
		data >>= 2,	pos += 2;
	if ((data & 0x1) == 0)
		pos += 1;
	return pos;
}

static inline size_t allign_size(size_t size){
  size_t ret = 1;
  while(ret < size)
    ret <<= 1;
  return ret;
}

#ifdef TEST
int check_page(int cpu){
  spin_lock(&fast_slab_manager[cpu].lock);
  int cnt = 0;
  for(int i = 0 ;i < FAST_SLAB_TYPE_NUM ;++i){
    page_t* cur = fast_slab_manager[cpu].head[i];
    while(cur != NULL){
      ++cnt;
      cur = cur->next;
    }
  }
  spin_unlock(&fast_slab_manager[cpu].lock);
  return cnt;
}

int check_free_list(){
  spin_lock(&fast_page_lock);
  int cnt = 0;
  page_t *cur = free_page;
  while(cur != NULL){
    cnt ++;
    cur = cur->next;
  }
  spin_unlock(&fast_page_lock);
  return cnt;
}

#endif

/* slab manager */
static int alloc_slab(int id){ // 1 if fast, 2 if slow
  spin_lock(&slab_lock);
  int i = 0;
  for(;i < SLAB_NUM ;++i)
    if(slab_manager[i].used == 0){
      slab_manager[i].used = id; break;
    }
  if(i == SLAB_NUM) i = -1;
  spin_unlock(&slab_lock);
  return i;
}

static int fast_new_slab(){
  assert(free_page == NULL);
  int id = alloc_slab(FAST);
  if(id == -1) return -1;
  free_page = slab_manager[id].start;
  page_t* tmp = free_page;
  while(tmp < (page_t*)((char*)free_page + SLAB_SIZE)){
    spin_init(&tmp->lock);
    if(tmp > free_page) tmp->prev = tmp - 1;
    if(tmp + 1 < (page_t*)((char*)free_page + SLAB_SIZE)) tmp->next = tmp + 1;
    tmp += 1;
  }
  (tmp - 1)->next = NULL;
  return 1;
}

static int slow_new_slab(int cpu){
  int id = alloc_slab(SLOW);
  if(id == -1) return -1;
  spin_lock(&slow_slab_lock);
  ++ SLOW_SLAB_NUM;
  slow_slab_manager[SLOW_SLAB_NUM - 1].start = slab_manager[id].start;
  #ifdef TEST
  memset(slow_slab_manager[SLOW_SLAB_NUM - 1].start, 0, SLOW_SLAB_SIZE);
  memset(&slow_slab_manager[SLOW_SLAB_NUM - 1].map, 0, sizeof(slow_slab_manager[SLOW_SLAB_NUM - 1].map));
  #endif
  spin_init(&slow_slab_manager[SLOW_SLAB_NUM - 1].lock);
  slow_slab_manager[SLOW_SLAB_NUM - 1].valid = 1;
  slow_slab_manager[SLOW_SLAB_NUM - 1].cpu = cpu;
  int ret = SLOW_SLAB_NUM - 1;
  spin_unlock(&slow_slab_lock);
  return ret;
}

/* page manager */
static page_t* alloc_fast_page(size_t size, int cpu){
  spin_lock(&fast_page_lock);
  if(free_page == NULL)
    fast_new_slab();
  if(free_page){
    page_t *ret = free_page;
    free_page = free_page->next;
    spin_unlock(&fast_page_lock);
    /* init new page */
    memset((void*)ret, 0, HDR_SIZE);
    spin_init(&ret->lock);
    ret->cpu = cpu;
    ret->obj_size = size;
    ret->obj_cnt = 0;
    ret->next = ret->prev = NULL;
    if(size <= 256)
      ret->obj_start = (void*)ret->data;
    else
      ret->obj_start = (void*)((char*)ret + size);
    ret->bit_num = ((uintptr_t)(ret + 1) - (uintptr_t)ret->obj_start) / size;
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
    spin_lock(&cur_page->lock);
    if(cur_page->obj_cnt < cur_page->bit_num)
      for(int i = 0 ;i < (cur_page->bit_num + 31) / 32 ;++i){
        if(cur_page->bitmap[i] == FULL) continue;
        int t = first_zero_bit(cur_page->bitmap[i]);
        if(i * 32 + t + 1 > cur_page->bit_num) break;
        cur_page->bitmap[i] |= (1 << t);
        cur_page->obj_cnt ++;
        spin_unlock(&cur_page->lock);
        return (char*)cur_page->obj_start + (i * 32 + t) * size;
      }
    else {
      spin_unlock(&cur_page->lock);
      cur_page = cur_page->next;
    }
  }
  return NULL;
}

static void fast_free(void* ptr){
  page_t *page = (page_t*)((char*)ptr - (uintptr_t)ptr % PAGE_SIZE);
  spin_lock(&page->lock);
  int cpu = page->cpu;
  int free = 0;
  int idx = (int)((char*)ptr - (char*)page->obj_start) / page->obj_size;
  int i = idx / 32;
  if(page->bitmap[i] & (1 << (idx % 32))){
    page->bitmap[i] &= ~(1 << (idx % 32));
    page->obj_cnt --;
    if(page->obj_cnt == 0) free = 1;
  }
  spin_unlock(&page->lock);
  if(free){
    spin_lock(&fast_slab_manager[cpu].lock);
    spin_lock(&page->lock);
    if(page->obj_cnt == 0)
      free_fast_page(page);
    spin_unlock(&page->lock);
    spin_unlock(&fast_slab_manager[cpu].lock);
  }
}

/* slow path */
static void* slow_alloc(size_t size, int id){
  spin_lock(&slow_slab_manager[id].lock);
  char* start = slow_slab_manager[id].start;
  assert(slow_slab_manager[id].valid);
  int map_cnt = size / (32 KB);
  for(int i = 0 ;i < SLOW_SLAB_SIZE / PAGE_SIZE ;i += MAX(map_cnt, slow_slab_manager[id].map[i])){
    int valid = 1;
    for(int j = i ;j < i + map_cnt ; ++j)
      if(slow_slab_manager[id].map[j]){
        valid = 0;
        break;
      }
    if(!valid) continue;
    // for(int j = i ;j < i + map_cnt ; ++j)
    //   slow_slab_manager[id].map[j] = map_cnt - (j - i);
    slow_slab_manager[id].map[i] = map_cnt;
    spin_unlock(&slow_slab_manager[id].lock); 
    return start + i * (32 KB);
  }
  spin_unlock(&slow_slab_manager[id].lock);
  return NULL;
}

static void slow_free(void* ptr, int id){
  spin_lock(&slow_slab_manager[id].lock);
  int start_idx = (int)((char*)ptr - (char*)slow_slab_manager[id].start) / PAGE_SIZE;
  // int map_cnt = slow_slab_manager[id].map[start_idx];
  // for(int i = start_idx ;i < start_idx + map_cnt; ++i)
  //   slow_slab_manager[id].map[i] = 0;
  slow_slab_manager[id].map[start_idx] = 0;
  spin_unlock(&slow_slab_manager[id].lock);
}

/* kalloc, kfree, pmm_init */
static void *kalloc(size_t size) {
  if (size > 16 MB) return NULL;
  size = MAX(allign_size(size), 8);
  int cpu = cpu_current();
  if (size <= 16 KB){  // fast path
    spin_lock(&fast_slab_manager[cpu].lock); // LOCK
    void* ret = fast_alloc(size, cpu);
    // spin_unlock(&fast_slab_manager[cpu].lock); // UNLOCK
    if(!ret) {
      page_t* new_page = alloc_fast_page(size, cpu);
      if(new_page == NULL) return NULL;
      // spin_lock(&fast_slab_manager[cpu].lock);  // LOCK
      page_t* head = fast_slab_manager[cpu].head[size2idx(size)];
      new_page->next = head;
      if(head) head->prev = new_page;
      fast_slab_manager[cpu].head[size2idx(size)] = new_page;
      ret = fast_alloc(size, cpu);
      // spin_unlock(&fast_slab_manager[cpu].lock);  // UNLOCK
      assert(ret);
    }
    spin_unlock(&fast_slab_manager[cpu].lock);
    return ret;
  } else{  // slow path
    int id = rand() % MAX_SLOW_SLAB_NUM;
    for(int i = 0 ;i < MAX_SLOW_SLAB_NUM ;++i)
      if(slow_slab_manager[(id + i) % MAX_SLOW_SLAB_NUM].valid 
        && slow_slab_manager[(id + i) % MAX_SLOW_SLAB_NUM].cpu == cpu){
        void *ret = slow_alloc(size, (id + i) % MAX_SLOW_SLAB_NUM);
        if(ret) return ret;
      }
    int new_slab_id = slow_new_slab(cpu);
    if(new_slab_id != -1){ 
      void *ret = slow_alloc(size, new_slab_id);
      assert(ret);
      return ret;
    }
    return NULL;
  }  
}

static void kfree(void *ptr) {
  if((char*)ptr < (char*)heap_start || (char*)ptr >= (char*)heap_end) return;
  int id = ((uintptr_t)ptr - (uintptr_t)ptr % SLAB_SIZE - (uintptr_t)heap_start) / SLAB_SIZE;
  int status = slab_manager[id].used;
  if(status == FAST){
    fast_free(ptr);
  } else if (status == SLOW) {
    for(int i = 0 ;i < MAX_SLOW_SLAB_NUM ;++i)
      if(slab_manager[id].start == slow_slab_manager[i].start){
        slow_free(ptr, i);
        return;
      }
    assert(0);
  } else
    assert(0);
}

static void pmm_init() {
  #ifndef TEST
  uintptr_t pmsize = ((uintptr_t)heap.end - (uintptr_t)heap.start);
  printf("Got %d MiB heap: [%p, %p)\n", pmsize >> 20, heap.start, heap.end);
  #else
  heap.start = malloc(HEAP_SIZE);
  heap.end   = (char*)heap.start + HEAP_SIZE;
  printf("Got %d MiB heap: [%p, %p)\n", HEAP_SIZE >> 20, heap.start, heap.end);
  spin_init(&cpu_lock);
  #endif
  /* allign heap size */
  if((uintptr_t)heap.start % SLAB_SIZE != 0)
    heap_start = (void*)((uintptr_t)heap.start + SLAB_SIZE - (uintptr_t)heap.start % (SLAB_SIZE));
  heap_end = (void*)((uintptr_t)heap.end - (uintptr_t)heap.end % (SLAB_SIZE));
  printf("Alligned heap_start: %p, heap_end: %p\n", heap_start, heap_end);
  /* init slab */
  spin_init(&slab_lock);
  SLAB_NUM = ((char*)heap_end - (char*)heap_start) / SLAB_SIZE;
  for(int i = 0 ;i < SLAB_NUM ;++i)
    slab_manager[i].start = (void*)((char*)heap_start + i * SLAB_SIZE);
  assert(SLAB_NUM >= 2);
  /* init fast & slow page */
  spin_init(&slow_slab_lock);
  spin_init(&fast_page_lock);
}

MODULE_DEF(pmm) = {
  .init  = pmm_init,
  .alloc = kalloc,
  .free  = kfree,
};

