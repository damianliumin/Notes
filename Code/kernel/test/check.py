import sys

testid = int(sys.argv[1])


def check_addr_align(size, addr, cur_idx, cpu):
    i = 1
    while i < size:
        i <<= 1
    if addr % i != 0:
        print("problem detected in #%d-(%d), address not alligned: %x"%(cur_idx, cpu, addr))
        print("fail in test%d" % (testid))
        exit()


def check_new_space(addr_rec, new_space, cur_idx, cpu):
    for space in addr_rec:
        if (new_space[0] < space[0] and new_space[1] > space[0]) or \
            (new_space[0] >= space[0] and new_space[0] < space[1]):
            print("problem detected in #%d-(%d), conflict with #%d-(%d)" % (cur_idx, cpu, space[2], space[3]))
            print("fail in test%d" % (testid))
            exit()


def check_prev_ptr(addr_rec, ptr, cur_idx, cpu):
    for space in addr_rec:
        if space[0] == ptr:
            addr_rec.remove(space)
            return
    print("problem detected in #%d-(%d), ptr %x not found"%(cur_idx, cpu, ptr))
    exit()


if __name__ == "__main__":
    with open('test/log/log_test{}.txt'.format(testid), 'r') as fp:
        print("checking test{}...".format(testid))
        addr_rec = []
        # skip init info
        for line in fp:
            if line == "#test{}\n".format(testid):
                break
        # check log
        succ, fail = 0, 0
        for line in fp:
            if line[0] != '#':
                continue
            str_ele = line.split()
            i = int(str_ele[0][1:])
            if str_ele[1] == 'kalloc':  # kalloc
                if str_ele[3] == 'fail':
                    fail = fail + 1
                else:
                    cpu = int(str_ele[5])
                    new_space = (int(str_ele[3], 16), int(str_ele[4], 16), i, cpu)
                    check_addr_align(int(str_ele[2]), new_space[0], i, cpu) # check addr align
                    check_new_space(addr_rec, new_space, i, cpu) # check conflicts with prev mem
                    addr_rec.append(new_space) # safe, append to record
                    succ = succ + 1
            elif str_ele[1] == 'kfree':  # kfree
                ptr = int(str_ele[2], 16)
                cpu = int(str_ele[3])
                check_prev_ptr(addr_rec, ptr, i, cpu)


        print("succ %d fail %d"%(succ, fail))
        print("pass test{}".format(testid))
        
