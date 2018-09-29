NAME = cuda-spgram

# nvcc
NVCC      := nvcc
NVCCFLAGS :=
NVCCFLAGS += -G
NVCCFLAGS += --compiler-options '-fPIC'
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
	

cuda-spgram-cf.o : cuda-spgram-cf.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
	
	
clean:
	rm -f lib$(NAME).a lib$(NAME).so cuda-spgram-cf.o