
#include <cuda-spgram-cf.h>

int main(int argc, char* argv[]) {

  CudaSpGramCF* q = CudaSpGramCF::create_default(1024);
  q->print();
  delete q;
}
