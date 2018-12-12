// Copyright [2018] <Alexander Hurd>

#ifndef CUDA_SPGRAM_CF_HPP_
#define CUDA_SPGRAM_CF_HPP_

// cuda cufft for cufftComplex
#include <cufft.h>
#include <complex>

//  thrust includes that use kernels below
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

struct clear_cufftComplex {
  __host__    __device__ cufftComplex operator()() {
    cufftComplex s;
    s.x = 0.0f;
    s.y = 0.0f;
    return s;
  }
};

struct apply_window {
  __host__  __device__ cufftComplex operator()(std::complex<float> s, float w) {
    //   printf("%.6f - %.6f\n", s.real(), s.imag());
    cufftComplex v;
    v.x = s.real() * w;
    v.y = s.imag() * w;
    //  printf("%.6f - %.6f : w  %.6f\n", s.real(), s.imag(), w);
//   printf("%.6f - %.6f : w  %.6f\n", s.real(), s.imag(), w);
    //  printf("%.6f - %.6f\n", v.x, v.y);
    return v;
  }
};

struct printf_complex {
  __host__ __device__ void operator()(std::complex<float> s) {
    printf("%.6f - %.6f\n", s.real(), s.imag());
  }
};

struct printf_float {
  __host__ __device__ void operator()(float s) {
    printf("%.6f\n", s);
  }
};

struct printf_cufftComplex {
  __host__ __device__ void operator()(cufftComplex s) {
    printf("%.6f - %.6f\n", s.x, s.y);
  }
};


#endif /* CUDA_SPGRAM_CF_HPP_ */
