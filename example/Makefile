CFLAGS = -std=c++11 -lstdc++ -Wall -Werror -I../ -L../ -L=/usr/local/cuda/targets/x86_64-linux/lib/

all:
	$(CXX) $(CFLAGS) example.cc -o example -lcuda-spgram -lcufft -lcudart