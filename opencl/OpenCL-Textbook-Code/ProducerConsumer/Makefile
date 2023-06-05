# OpenCL include directory
OCL_INC = /opt/AMDAPPSDK-3.0/include

# OpenCL lib directory
OCL_LIB = /opt/AMDAPPSDK-3.0/lib/x86_64

# C compiler to use
CC = gcc

# compile-time flags
CFLAGS = -Wall -O3

# header file directories
INCLUDES = -I/usr/include -I${OCL_INC} -I../../Utils

# library paths
LFLAGS = -L${OCL_LIB}

# libraries
LIBS = -lOpenCL

# source files
SRCS = producer-consumer.c ../../Utils/utils.c ../../Utils/bmp-utils.c ../../Utils/gold.c

# object files 
OBJS = $(SRCS:.c=.o)

# executable file 
MAIN = producer-consumer 

.PHONY: clean

all:    $(MAIN)

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)
