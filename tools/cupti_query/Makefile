#
# Copyright 2011-2018 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I/usr/local/cuda/extra/CUPTI/include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    LIB_PATH ?= /usr/local/cuda/extra/CUPTI/lib64
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib64
        LIBS= -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
endif

cupti_query: cupti_query.$(OBJ)
	nvcc -o $@ cupti_query.$(OBJ) $(LIBS)

cupti_query.$(OBJ): cupti_query.cpp
	nvcc -c $(INCLUDES) $<

run: cupti_query
	./$<

clean:
	rm -f cupti_query cupti_query.$(OBJ)
