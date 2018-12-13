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

struct clear_float {
  __host__   __device__ float operator()() {
    return 0.0f;
  }
};

struct apply_window {
  __host__  __device__ cufftComplex operator()(std::complex<float> s, float w) {
    cufftComplex v;
    v.x = s.real() * w;
    v.y = s.imag() * w;
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

struct first_psd {
  cufftComplex* source_;
  float* psd_;
  first_psd(cufftComplex* source, float* psd):
    source_(source), psd_(psd) {}
  __host__  __device__ void operator()(const uint64_t i) {
    cufftComplex s = source_[i];
    float v = (s.x * s.x) + (s.y * s.y);
    psd_[i] = v;
  }
};

struct accumulate_psd {
  cufftComplex* source_;
  float* psd_;
  float alpha_, gamma_;
  accumulate_psd(cufftComplex* source, float* psd, float alpha, float gamma):
    source_(source), psd_(psd), alpha_(alpha), gamma_(gamma) {}
  __host__  __device__ void operator()(const uint64_t i) {
    cufftComplex s = source_[i];
    float v = (s.x * s.x) + (s.y * s.y);
    psd_[i] = gamma_ * psd_[i] + alpha_ * v;
  }
};

struct calc_power_and_shift {
  float* psd_;
  float* out_;
  float scale_;
  uint32_t nfft_;
  calc_power_and_shift(float* psd, float* out, float scale, uint32_t nfft):
    psd_(psd), out_(out), scale_(scale), nfft_(nfft) {}
  __host__  __device__ void operator()(const uint64_t i) {
    uint32_t nfft_2 = nfft_ / 2;
    uint32_t k = (i + nfft_2) % nfft_;
    out_[i] =  10 * log10f(psd_[k] + 1e-6f) + scale_;
  }
};

#endif /* CUDA_SPGRAM_CF_HPP_ */
