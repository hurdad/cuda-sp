// Copyright [2018] <Alexander Hurd>"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <complex>
#include <liquid/liquid.h>

#ifndef CUDA_SPGRAM_CF_H_
#define CUDA_SPGRAM_CF_H_

class CudaSpGramCF {
 public:
  CudaSpGramCF();
  virtual ~CudaSpGramCF();
  static CudaSpGramCF* create(unsigned int _nfft, int _wtype, unsigned int _window_len, unsigned int _delay);
  static CudaSpGramCF* create_default(unsigned int _nfft);
  void clear();
  void reset();
  inline void print() {
    printf("CudaSpGramCF: nfft=%u, window=%u, delay=%u\n", nfft, window_len, delay);
  }
  inline void set_alpha(float _alpha) {
    alpha = _alpha;
  }
  inline void set_freq(float _freq) {
    frequency = _freq;
  }
  inline void set_rate(float _rate) {
    sample_rate = _rate;
  }
  inline size_t get_nfft() {
    return nfft;
  }
  inline size_t get_window_len() {
    return window_len;
  }
  inline size_t get_delay() {
    return delay;
  }
  inline uint64_t get_num_samples() {
    return num_samples;
  }
  inline uint64_t get_num_samples_total() {
    return num_samples_total;
  }
  inline uint64_t get_num_transforms() {
    return num_samples_total;
  }
  inline uint64_t get_num_transforms_total() {
    return num_transforms_total;
  }
  void push();
  void write();
  void step();
  void get_psd();
  void export_gnuplot( const char* _filename);
  void estimate_psd();

 private:
  // options
  unsigned int    nfft;           // FFT length
  int             wtype;          // window type
  unsigned int    window_len;     // window length
  unsigned int    delay;          // delay between transforms [samples]
  float           alpha;          // spectrum smoothing filter: feedforward parameter
  float           gamma;          // spectrum smoothing filter: feedback parameter
  int             accumulate;     // accumulate? or use time-average
  /*
  	WINDOW()        buffer;         // input buffer
  	TC *            buf_time;       // pointer to input array (allocated)
  	TC *            buf_freq;       // output fft (allocated)
  	T  *            w;              // tapering window [size: window_len x 1]
  	FFT_PLAN        fft;            // FFT plan
  */
  // psd accumulation
//	T *             psd;                    // accumulated power spectral density estimate (linear)
  unsigned int    sample_timer;           // countdown to transform
  uint64_t        num_samples;            // total number of samples since reset
  uint64_t        num_samples_total;      // total number of samples since start
  uint64_t        num_transforms;         // total number of transforms since reset
  uint64_t        num_transforms_total;   // total number of transforms since start

  // parameters for display purposes only
  float           frequency;      // center frequency [Hz]
  float           sample_rate;    // sample rate [Hz]
};

#endif /* CUDA_SPGRAM_CF_H_ */
