NAME = cuda-sp
DESTDIR = /usr/local

# nvcc
NVCC      := nvcc
NVCCFLAGS += -G -g -m64
NVCCFLAGS += --compiler-options '-O2 -fPIC -I/usr/local/cuda/samples/common/inc' --expt-relaxed-constexpr
NVCCFLAGS += -gencode arch=compute_30,code=sm_30 

LIB_BUILD  = lib$(NAME).so
LIBLINKER = $(CXX)
LIB_LFLAGS = -shared -o

all: library

library: $(LIB_BUILD)

$(LIB_BUILD): cuda-spgram-cf.o cuda-spwaterfall-cf.o
	$(LIBLINKER) $(LIB_LFLAGS) $(LIB_BUILD) cuda-spgram-cf.o cuda-spwaterfall-cf.o
	
cuda-spgram-cf.o: cuda-spgram-cf.cu cuda-spgram-cf.h
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
	
cuda-spwaterfall-cf.o: cuda-spwaterfall-cf.cc cuda-spwaterfall-cf.h
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
  	
example: library
	$(MAKE) -C example/
	LD_LIBRARY_PATH=./ ./example/spgram-example
	LD_LIBRARY_PATH=./ ./example/spwaterfall-example
	
install: library
	mkdir -p $(DESTDIR)/include/
	cp cuda-spgram-cf.h $(DESTDIR)/include/
	cp cuda-spwaterfall-cf.h $(DESTDIR)/include/
	mkdir -p $(DESTDIR)/lib64/
	cp $(LIB_BUILD) $(DESTDIR)/lib64/
	
clean:
	rm -f lib$(NAME).so cuda-spgram-cf.o cuda-spwaterfall-cf.o example/spgram-example *.gnu *.bin *.png
	
format:
	astyle --options=astyle.options cuda-spgram-cf.cu cuda-spgram-cf.h cuda-spwaterfall-cf.cc cuda-spwaterfall-cf.h example/spgram-example.cc
	
rpm-build:
	cd rpm && make
