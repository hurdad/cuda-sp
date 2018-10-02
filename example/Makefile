CFLAGS = -std=c++11 -g -lstdc++ -Wall -Werror -I../ -I/usr/local/cuda/targets/x86_64-linux/include/ -I/usr/local/cuda/samples/common/inc -L../ -L/usr/local/cuda/targets/x86_64-linux/lib/

all:
	$(CXX) $(CFLAGS) example.cc -o example -lcuda-spgram -lcufft -lcudart -lliquid