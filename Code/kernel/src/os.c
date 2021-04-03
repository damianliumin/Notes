#include <common.h>

static void os_init() {
  pmm->init();
}


static void os_run() {
  #ifndef TEST
  for (const char *s = "Hello World from CPU #*\n"; *s; s++) {
    putch(*s == '*' ? '0' + cpu_current() : *s);
  }

  for(int i = 1 ;i <= 100 ;++i){
    size_t size = (rand() % (1024 KB));
    printf("size: %d MB %d KB %d B\n", size / (1 MB), size % (1 MB) / (1 KB), size / (1 KB));
    void *ptr = pmm->alloc(size);
    if(ptr == NULL)
      printf("#%d: Fail to alloc %dB\n", i, size);
    else 
      printf("#%d: [%p, %p), page %d, size %d\n", i, ptr, (char*)ptr + size, 
            ((char*)ptr - (char*)heap.start) / PAGE_SIZE, size);
  }  
  #endif
  while (1) ;
}

MODULE_DEF(os) = {
  .init = os_init,
  .run  = os_run,
};
