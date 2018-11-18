// Copyright [2018] <Alexander Hurd>

#ifndef CUDA_SPGRAM_CF_H_
#define CUDA_SPGRAM_CF_H_

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <vector>

// liquid dsp
#include <complex>
#include <liquid/liquid.h>

// cuda cufft for cufftComplex
#include <cufft.h>

class CudaSpGramCF {
 public:
  // cuda memory api to use
  enum CudaMemoryAPI_t { DEVICE_MAPPED, UNIFIED};

  // create CudaSpGramCF object
  //  _nfft       : FFT size
  //  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING
  //  _window_len : window length
  //  _delay      : delay between transforms, _delay > 0
  static CudaSpGramCF* create(unsigned int _nfft,
                              int          _wtype,
                              unsigned int _window_len,
                              unsigned int _delay,
                              CudaMemoryAPI_t _api);

  // create default CudaSpGramCF object (Kaiser-Bessel window)
  static CudaSpGramCF* create_default(unsigned int _nfft);

  // destroy CudaSpGramCF object
  virtual ~CudaSpGramCF();

  // clears the internal state of the CudaSpGramCF object, but not
  // the internal buffer
  void clear();

  // reset the CudaSpGramCF object to its original state completely
  void reset();

  // prints the CudaSpGramCF object's parameters
  void print();

  // set forgetting factor
  int set_alpha(float _alpha);

  // set center frequency
  void set_freq(float _freq);

  // set sample rate
  int set_rate(float _rate);

  // get FFT size
  size_t get_nfft();

  // get window length
  size_t get_window_len();

  // get delay between transforms
  size_t get_delay();

  // get number of samples processed since reset
  uint64_t get_num_samples() ;

  // get number of samples processed since start
  uint64_t get_num_samples_total();

  // get number of transforms processed since reset
  uint64_t get_num_transforms();

  // get number of transforms processed since start
  uint64_t get_num_transforms_total();

  /// push a single sample into the CudaSpGramCF object
  //  _x      :   input sample
  void push(liquid_float_complex _x);

  // write a block of samples to the CudaSpGramCF object
  //  _x      :   input buffer [size: _n x 1]
  //  _n      :   input buffer length
  void write(liquid_float_complex* _x,
             size_t _n);

  // compute spectral periodogram output from current buffer contents
  void step();

  // compute spectral periodogram output (fft-shifted values
  // in dB) from current buffer contents
  //  _X      :   output spectrum [size: _nfft x 1]
  void get_psd(float* _X);

  // export gnuplot file
  //  _filename : input buffer [size: _n x 1]
  int export_gnuplot(const char* _filename);

  // estimate spectrum on input signal
  //  _nfft   :   FFT size
  //  _x      :   input signal [size: _n x 1]
  //  _n      :   input signal length
  //  _psd    :   output spectrum, [size: _nfft x 1]
  static void estimate_psd(unsigned int _nfft,
                           liquid_float_complex* _x,
                           unsigned int _n,
                           float* _psd);

 private:
  // options
  unsigned int    nfft;           // FFT length
  int             wtype;          // window type
  unsigned int    window_len;     // window length
  unsigned int    delay;          // delay between transforms [samples]
  float           alpha;          // spectrum smoothing filter: feedforward parameter
  float           gamma;          // spectrum smoothing filter: feedback parameter
  int             accumulate;     // accumulate? or use time-average
  CudaMemoryAPI_t api;			  // cuda memory api to use

  windowcf      			buffer;     // input buffer
  cufftComplex* 			buf_time;   // pointer to input array (allocated)
  cufftComplex* 			d_buf_time; // pointer to input device array (allocated)
  cufftComplex*  			buf_freq;   // output fft (allocated)
  std::vector<float>        w;          // tapering window [size: window_len x 1]
  cufftHandle 				fft;		// FFT plan

  // psd accumulation
  std::vector<float> psd;   			  // accumulated power spectral density estimate (linear)
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
