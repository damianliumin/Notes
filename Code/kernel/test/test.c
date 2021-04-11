#include <common.h>
#include <threads.h>
#include <pthread.h>
#define TEST_NUM 2000
#define TEST_THREAD_NUM 8
#define ALLOC_RATE 90

static void entry(int tid) { for(int i = 1 ;i <= 10 ;++i) pmm->alloc(128); }
static void goodbye()      { printf("End.\n"); }

void *ptr_rec[TEST_THREAD_NUM][TEST_NUM + 5] = {0};
int ptr_num[TEST_THREAD_NUM] = {0};
// int rec_i[TEST_THREAD_NUM] = {0};

static inline size_t get_size(){
    // 75% <= 128 B
    // 10% 128 B ~ 4 KB
    // 10% 4KB ~ 64KB
    // 5% 64KB ~ 1M
    // int tmp = rand() % 99 + 1;
    // if(tmp <= 75)
    //     return rand() % 128 + 1;
    // else if(tmp <= 85)
    //     return rand() % (4 KB - 128) + 1 + 128;
    // else if(tmp <= 95)
    //     return rand() % (64 KB - 4 KB) + 1 + 4 KB;
    // else
    //     return rand() % (1 MB - 64 KB) + 1 + 64 KB;    
    return 4 KB;
}

static inline int get_action(int per){ // return 0 if alloc, 1 if free
    assert(per >= 0 && per <= 100);
    int tmp = rand() % 101;
    return tmp > per;
}

int cpu_current();

void test(){
    int cpu = cpu_current();
    uint32_t big_size = 0;
    for(int i = 0 ;i < TEST_NUM ;++i){
        // get action
        int action = get_action(ALLOC_RATE);
        if(action == 0 || i == 0 || ptr_num[cpu] == 0) {  // alloc
            size_t size = get_size();
            if(size >= 4 KB)
                big_size += size;
            void* ptr = pmm->alloc(size);
            if(ptr){
                ptr_rec[cpu][i] = ptr;
                ++ ptr_num[cpu];
                printf("#%d kalloc\t%ld\t%p\t%p\t%d\n", i, size, ptr, (char*)ptr +size, cpu);
            } else
                printf("#%d kalloc\t%ld\tfail\t%d\n", i, size, cpu);
        } else {  // free
            int idx = rand() % i;
            int j = 0;
            for(;j < i ;++j)
                if(ptr_rec[cpu][(idx + j) % i]){
                    printf("#%d kfree\t%p\t%d\n", i, ptr_rec[cpu][(idx + j) % i], cpu);
                    pmm->free(ptr_rec[cpu][(idx + j) % i]);
                    ptr_rec[cpu][(idx + j) % i] = NULL;
                    -- ptr_num[cpu];
                    break;
                }
            assert(j < i);
        }
    }

    int check_page(int);
    int check_free_list();
    printf("big size: %u MB %u KB\n", big_size / (1 MB), big_size / (1 KB));
    for(int i = 0 ;i < TEST_THREAD_NUM ;++i)
        printf("check page: %d\n", check_page(i));
    printf("check free: %d\n", check_free_list());
    extern int SLOW_SLAB_NUM;
    extern kheap_slab_t slab_manager[MAX_SLAB_NUM];
    int fast_slab_num = 0;
    for(int i = 0 ;i < MAX_SLAB_NUM; ++i)
        if(slab_manager[i].used == FAST)
            ++fast_slab_num;
    printf("fast slab num: %d\n", fast_slab_num);
    printf("slow slab num: %d\n", SLOW_SLAB_NUM);
}


static void* ptr_rec2[TEST_THREAD_NUM * TEST_NUM + 10] = {0};
static int ptr_cnt;
static spinlock_t rec_lock;

void test2(){
    int cpu = cpu_current();
    for(int i = 0 ;i < TEST_NUM ;++i){
        int action = get_action(ALLOC_RATE);
        spin_lock(&rec_lock);
        int tmp_cnt = ptr_cnt;
        spin_unlock(&rec_lock);
        if(action == 0 || tmp_cnt == 0){
            size_t size = get_size();
            void* ptr = pmm->alloc(size);
            if(ptr){
                printf("#%d kalloc\t%ld\t%p\t%p\t%d\n", i, size, ptr, (char*)ptr +size, cpu);
                spin_lock(&rec_lock);
                ptr_rec2[++ptr_cnt] = ptr;
                spin_unlock(&rec_lock);
            } else {
                printf("#%d kalloc\t%ld\tfail\t%d\n", i, size, cpu);
            }
        } else {
            void* ptr = NULL;
            spin_lock(&rec_lock);
            if(ptr_cnt)
                ptr = ptr_rec2[ptr_cnt --];
            spin_unlock(&rec_lock);
            if(ptr){
                printf("#%d kfree\t%p\t%d\n", i, ptr, cpu);
                pmm->free(ptr);
            }                
        }
    }
    int check_page(int);
    int check_free_list();
    for(int i = 0 ;i < TEST_THREAD_NUM ;++i)
        printf("check page: %d\n", check_page(i));
    printf("check free: %d\n", check_free_list());
    extern int SLOW_SLAB_NUM;
    extern kheap_slab_t slab_manager[MAX_SLAB_NUM];
    int fast_slab_num = 0;
    for(int i = 0 ;i < MAX_SLAB_NUM; ++i)
        if(slab_manager[i].used == FAST)
            ++fast_slab_num;
    printf("fast slab num: %d\n", fast_slab_num);
    printf("slow slab num: %d\n", SLOW_SLAB_NUM);
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

void do_test_2(){
    printf("#test2\n");
    for(int i = 0 ;i < TEST_THREAD_NUM ;++i)
        create(test2);
    join(NULL);
}

int main(int argc, char *argv[]) {
    pmm->init();
    srand(time(NULL));
    int testid = atoi(argv[1]);
    switch(testid){
        case 0: do_test_0(); break;
        case 1: do_test_1(); break;
        case 2: do_test_2(); break;
        default: assert(0);
    }
    
    return 0;
}