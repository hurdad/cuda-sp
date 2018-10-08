// Copyright [2018] <Alexander Hurd>

#ifndef CUDA_SPWATERFALL_CF_H_
#define CUDA_SPWATERFALL_CF_H_

#include <stdlib.h>
#include <string.h>

#include "cuda-spgram-cf.h"

class CudaSpWaterfallCF {
 public:
  // create CudaSpWaterfallCF object
  //  _nfft       : FFT size
  //  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING
  //  _window_len : window length
  //  _delay      : delay between transforms, _delay > 0
  static CudaSpWaterfallCF* create(unsigned int _nfft,
                                   int          _wtype,
                                   unsigned int _window_len,
                                   unsigned int _delay,
                                   unsigned int _time);

  // create default spwaterfall object (Kaiser-Bessel window)
  static CudaSpWaterfallCF* create_default(unsigned int _nfft, unsigned int _time);

  // destroy CudaSpWaterfallCF object
  virtual ~CudaSpWaterfallCF();

  // clears the internal state of the CudaSpWaterfallCF object, but not
  // the internal buffer
  void clear();

  // reset the CudaSpWaterfallCF object to its original state completely
  void reset();

  // prints the CudaSpWaterfallCF object's parameters
  void print();

  // set center frequency
  int set_freq(float _freq);

  // set sample rate
  int set_rate(float _rate);

  // set image dimensions
  int set_dims(unsigned int _width,
               unsigned int _height);

  // push a single sample into the CudaSpWaterfallCF object
  //  _x      :   input sample
  void push(liquid_float_complex _x);

  // write a block of samples to the CudaSpWaterfallCF object
  //  _x      :   input buffer [size: _n x 1]
  //  _n      :   input buffer length
  void write(liquid_float_complex* _x, size_t _n);

  // export output files
  //  _base : base filename
  int export_files(const char* _base);

  // compute spectral periodogram output from current buffer contents
  void step();

  // consolidate buffer by taking log-average of two separate spectral estimates in time
  void consolidate_buffer();

  // export binary spectral file
  //  _filename : input buffer [size: _n x 1]
  int export_bin(const char* _base);

  // export gnuplot file
  //  _filename : input buffer [size: _n x 1]
  int export_gnu(const char* _base);

 private:
  // options
  unsigned int    nfft;           // FFT length
  unsigned int    time;           // minimum time buffer
  CudaSpGramCF*    periodogram;   // spectral periodogram object

  // buffers
  std::vector<float>     psd;     // time/frequency buffer [nfft x 2*time]
  unsigned int    index_time;     // time index for writing to buffer
  unsigned int    rollover;       // number of FFTs to take before writing to output

  // parameters for display purposes only
  float           frequency;      // center frequency [Hz]
  float           sample_rate;    // sample rate [Hz]
  unsigned int    width;          // image width [pixels]
  unsigned int    height;         // image height [pixels]
};

#endif /* CUDA_SPWATERFALL_CF_H_ */
