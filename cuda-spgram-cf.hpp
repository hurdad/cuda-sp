// Copyright [2018] <Alexander Hurd>

#ifndef CUDA_SPGRAM_CF_HPP_
#define CUDA_SPGRAM_CF_HPP_

// cuda cufft for cufftComplex
#include <cufft.h>

//  thrust includes that use kernels below
#include <thrust/generate.h>
#include <thrust/device_ptr.h>

struct clear_cufftComplex {
	__host__   __device__ cufftComplex operator()() {
		cufftComplex s;
		s.x = 0.0f;
		s.y = 0.0f;
		return s;
	}
};


#endif /* CUDA_SPGRAM_CF_HPP_ */
