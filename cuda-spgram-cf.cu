#include "cuda-spgram-cf.h"


#include <cuda_runtime.h>
#include <cufft.h>

CudaSpGramCF::CudaSpGramCF(){

	cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * 1024);

	cufftComplex *d_signal;
	cudaMalloc((void **)&d_signal, 1024*sizeof(cufftComplex));

}

CudaSpGramCF::~CudaSpGramCF(){

}
