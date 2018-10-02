NAME = cuda-spgram

# nvcc
NVCC      := nvcc
NVCCFLAGS :=
NVCCFLAGS += -G -g
NVCCFLAGS += --compiler-options '-fPIC -I/usr/local/cuda/samples/common/inc'
NVCCFLAGS += -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_50,code=compute_50

LIB_BUILD  = lib$(NAME).so
LIBLINKER = $(CXX)
LIB_LFLAGS = -shared -o

all: library

library: $(LIB_BUILD)

$(LIB_BUILD): cuda-spgram-cf.o
	$(LIBLINKER) $(LIB_LFLAGS) $(LIB_BUILD) cuda-spgram-cf.o
	
cuda-spgram-cf.o: cuda-spgram-cf.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
  	
example: library
	$(MAKE) -C example/
	LD_LIBRARY_PATH=./ \
	./example/example
 
clean:
	rm -f lib$(NAME).so cuda-spgram-cf.o example/example
	
format:
	astyle --options=astyle.options cuda-spgram-cf.cu cuda-spgram-cf.h example/example.cc