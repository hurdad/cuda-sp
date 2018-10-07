NAME = cuda-sp

# nvcc
NVCC      := nvcc
NVCCFLAGS :=
NVCCFLAGS += -G -g -m64
NVCCFLAGS += --compiler-options '-O2 -fPIC -I/usr/local/cuda/samples/common/inc'
NVCCFLAGS += -gencode arch=compute_30,code=sm_30 

LIB_BUILD  = lib$(NAME).so
LIBLINKER = $(CXX)
LIB_LFLAGS = -shared -o

all: library

library: $(LIB_BUILD)

$(LIB_BUILD): cuda-spgram-cf.o cuda-spwaterfall-cf.o
	$(LIBLINKER) $(LIB_LFLAGS) $(LIB_BUILD) cuda-spgram-cf.o cuda-spwaterfall-cf.o
	
cuda-spgram-cf.o: cuda-spgram-cf.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
	
cuda-spwaterfall-cf.o: cuda-spwaterfall-cf.cc
	$(CXX) $< -c -o $@
  	
example: library
	$(MAKE) -C example/
	LD_LIBRARY_PATH=./ \
	./example/spgram-example
 
clean:
	rm -f lib$(NAME).so cuda-spgram-cf.o cuda-spwaterfall-cf.o example/example
	
format:
	astyle --options=astyle.options cuda-spgram-cf.cu cuda-spgram-cf.h cuda-spwaterfall-cf.cu cuda-spwaterfall-cf.h  example/example.cc