CFLAGS = -std=c++11 -O2 -lstdc++ -Wall -Werror -I../ -I/usr/local/cuda/targets/x86_64-linux/include/ -L../ -L/usr/local/cuda/targets/x86_64-linux/lib/

all:
	$(CXX) $(CFLAGS) spgram-example.cc -o spgram-example -lcuda-sp -lcudart -lcufft -lliquid
	$(CXX) $(CFLAGS) spwaterfall-example.cc -o spwaterfall-example -lcuda-sp -lcudart -lcufft -lliquid