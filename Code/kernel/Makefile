NAME           := kernel
SRCS           := framework/main.c $(shell find -L ./src/ -name "*.c")
INC_PATH       := include/ framework/

export AM_HOME := $(PWD)/../abstract-machine
ifeq ($(ARCH),)
export ARCH    := x86_64-qemu
endif

include $(AM_HOME)/Makefile
include ../Makefile.lab
image: git

test0: git
	@gcc $(shell find src/ -name "*.c")  \
		$(shell find test/ -name "*.c") \
		-Iframework -Itest -DTEST -lpthread \
		-o build/test
	@build/test 0 > test/log/log_test0.txt
	@python3 test/check.py 0

test1: git
	@gcc $(shell find src/ -name "*.c")  \
		$(shell find test/ -name "*.c") \
		-Iframework -Itest -DTEST -lpthread \
		-o build/test
	@build/test 1 > test/log/log_test1.txt
	@python3 test/check.py 1

test2: git
	@gcc $(shell find src/ -name "*.c")  \
		$(shell find test/ -name "*.c") \
		-Iframework -Itest -DTEST -lpthread \
		-o build/test
	@build/test 2 > test/log/log_test2.txt
	@python3 test/check.py 2
