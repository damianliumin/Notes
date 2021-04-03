#include <common.h>
#include <threads.h>
#include <pthread.h>
#define TEST_NUM 2000
#define TEST_THREAD_NUM 4

static void entry(int tid) { for(int i = 1 ;i <= 10 ;++i) pmm->alloc(128); }
static void goodbye()      { printf("End.\n"); }

static inline size_t get_size(){
    // 75% <= 128 B
    // 10% 128 B ~ 4 KB
    // 10% 4KB ~ 64KB
    // 5% 64KB ~ 1M
    int tmp = rand() % 99 + 1;
    if(tmp <= 75)
        return rand() % 128 + 1;
    else if(tmp <= 85)
    // else
        return rand() % (4 KB - 128) + 1 + 128;
    else if(tmp <= 95)
        return rand() % (64 KB - 4 KB) + 1 + 4 KB;
    else
        return rand() % (1 MB - 64 KB) + 1 + 64 KB;    
}

static inline int get_action(int per){ // return 0 if alloc, 1 if free
    assert(per >= 0 && per <= 99);
    int tmp = rand() % 100;
    return tmp > per;
}

int cpu_current();

void test(){
    int cpu = cpu_current();

    void *ptr_rec[TEST_NUM + 5] = {0};
    int ptr_num = 0;
    for(int i = 0 ;i < TEST_NUM ;++i){
        // get action
        int action = get_action(70);
        if(action == 0 || i == 0 || ptr_num == 0) {  // alloc
            size_t size = get_size();
            void* ptr = pmm->alloc(size);
            if(ptr){
                ptr_rec[i] = ptr;
                ++ ptr_num;
                printf("#%d kalloc\t%ld\t%p\t%p\t%d\n", i, size, ptr, (char*)ptr +size, cpu);
            } else
                printf("#%d kalloc\t%ld\tfail\t%d\n", i, size, cpu);
        } else {  // free
            int idx = rand() % i;
            int j = 0;
            for(;j < i ;++j)
                if(ptr_rec[(idx + j) % i]){
                    printf("#%d kfree\t%p\t%d\n", i, ptr_rec[(idx + j) % i], cpu);
                    pmm->free(ptr_rec[(idx + j) % i]);
                    ptr_rec[(idx + j) % i] = NULL;
                    -- ptr_num;
                    break;
                }
            assert(j < i);
        }
    }
}

void do_test_0(){
    printf("#test0\n");
    test();
}

void do_test_1(){
    printf("#test1\n");
    for(int i = 0 ;i < TEST_THREAD_NUM ;++i)
        create(test);
    join(NULL);
}

int main(int argc, char *argv[]) {
    pmm->init();
    srand(time(NULL));
    // for (int i = 0; i < 4; i++)
    //     create(entry);
    // join(goodbye);
    int testid = atoi(argv[1]);
    switch(testid){
        case 0: do_test_0(); break;
        case 1: do_test_1(); break;
        default: assert(0);
    }
    return 0;
}