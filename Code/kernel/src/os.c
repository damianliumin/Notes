#include <common.h>

static void os_init() {
  pmm->init();
}


static void os_run() {
  #ifndef TEST
  for (const char *s = "Hello World from CPU #*\n"; *s; s++) {
    putch(*s == '*' ? '0' + cpu_current() : *s);
  }

  // void* prev_ptr = NULL;
  for(int i = 1 ;i <= 3585 ;++i){
    // size_t size = rand() % (16 KB) + 1;
    size_t size = 4 KB;
    printf("size: %d PAGES %d KB %d\n", size / (4 KB), size / (1 KB), size / (32 KB));
    void *ptr = pmm->alloc(size);
    if(ptr == NULL)
      printf("#%d: Fail to alloc %dB\n", i, size);
    else 
      printf("#%d: [%p, %p), page %d, size %x\n", i, ptr, (char*)ptr + size, 
            ((char*)ptr - (char*)heap.start) / PAGE_SIZE, size);
  }  
  extern int SLAB_NUM;
  // extern kheap_slow_t slow_slab_manager[MAX_SLOW_SLAB_NUM];
  extern kheap_slab_t slab_manager[MAX_SLAB_NUM];

  printf("slab num: %d\n", SLAB_NUM);
  // for(int i = 0 ;i < MAX_SLAB_NUM ;++i)
  //   if(slow_slab_manager[i].valid){
  //     printf("#%d\n", i);
  //     for(int j = 0 ;j < SLOW_SLAB_SIZE / PAGE_SIZE ; ++j){
  //       printf("%d ", slow_slab_manager[i].map[j]);
  //     }
  //     printf("\n");
  //   }
  for(int i = 0 ;i < MAX_SLAB_NUM ;++i)
    if(slab_manager[i].used == FAST)
      printf("fast id: %d\n", i);
  #endif
  while (1) ;
}

MODULE_DEF(os) = {
  .init = os_init,
  .run  = os_run,
};
