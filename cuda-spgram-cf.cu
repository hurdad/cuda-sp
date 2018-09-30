// Copyright [2018] <Alexander Hurd>"

#include "cuda-spgram-cf.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

CudaSpGramCF::CudaSpGramCF() {

}

CudaSpGramCF::~CudaSpGramCF() {

}

CudaSpGramCF* CudaSpGramCF::create(unsigned int _nfft, int _wtype, unsigned int _window_len, unsigned int _delay) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpGramCFcreate(), fft size must be at least 2\n");
    exit(1);
  } else if (_window_len > _nfft) {
    fprintf(stderr, "error: CudaSpGramCF::create(), window size cannot exceed fft size\n");
    exit(1);
  } else if (_window_len == 0) {
    fprintf(stderr, "error: CudaSpGramCF::create(), window size must be greater than zero\n");
    exit(1);
  } else if (_wtype == LIQUID_WINDOW_KBD && _window_len % 2) {
    fprintf(stderr, "error: CudaSpGramCF::create(), KBD window length must be even\n");
    exit(1);
  } else if (_delay == 0) {
    fprintf(stderr, "error: CudaSpGramCF::create(), delay must be greater than 0\n");
    exit(1);
  }

  // allocate memory for main object
  CudaSpGramCF* q = new CudaSpGramCF();

  // set input parameters
  q->nfft       = _nfft;
  q->wtype      = _wtype;
  q->window_len = _window_len;
  q->delay      = _delay;
  q->frequency  =  0;
  q->sample_rate = -1;

  // set object for full accumulation
  q->set_alpha(-1.0f);


  cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * _nfft);
  cufftComplex* h_fft = (cufftComplex*)malloc(sizeof(cufftComplex) * _nfft);



  cufftComplex* d_signal;
  checkCudaErrors(cudaMalloc((void**)&d_signal, _nfft * sizeof(cufftComplex)));


  /*
  // create FFT arrays, object
  q->buf_time = (TC*) malloc((q->nfft) * sizeof(TC));
  q->buf_freq = (TC*) malloc((q->nfft) * sizeof(TC));
  q->psd      = (T*) malloc((q->nfft) * sizeof(T ));
  q->fft      = FFT_CREATE_PLAN(q->nfft, q->buf_time, q->buf_freq, FFT_DIR_FORWARD, FFT_METHOD);

  // create buffer
  q->buffer = WINDOW(_create)(q->window_len);

  // create window
  q->w = (T*) malloc((q->window_len) * sizeof(T));
  unsigned int i;
  unsigned int n = q->window_len;
  float beta = 10.0f;
  float zeta =  3.0f;
  for (i = 0; i < n; i++) {
    switch (q->wtype) {
    case LIQUID_WINDOW_HAMMING:
      q->w[i] = hamming(i, n);
      break;
    case LIQUID_WINDOW_HANN:
      q->w[i] = hann(i, n);
      break;
    case LIQUID_WINDOW_BLACKMANHARRIS:
      q->w[i] = blackmanharris(i, n);
      break;
    case LIQUID_WINDOW_BLACKMANHARRIS7:
      q->w[i] = blackmanharris7(i, n);
      break;
    case LIQUID_WINDOW_KAISER:
      q->w[i] = kaiser(i, n, beta, 0);
      break;
    case LIQUID_WINDOW_FLATTOP:
      q->w[i] = flattop(i, n);
      break;
    case LIQUID_WINDOW_TRIANGULAR:
      q->w[i] = triangular(i, n, n);
      break;
    case LIQUID_WINDOW_RCOSTAPER:
      q->w[i] = liquid_rcostaper_windowf(i, n / 3, n);
      break;
    case LIQUID_WINDOW_KBD:
      q->w[i] = liquid_kbd(i, n, zeta);
      break;
    default:
      fprintf(stderr, "error: spgram%s_create(), invalid window\n", EXTENSION);
      exit(1);
    }
  }

  // scale by window magnitude, FFT size
  float g = 0.0f;
  for (i = 0; i < q->window_len; i++)
    g += q->w[i] * q->w[i];
  g = M_SQRT2 / ( sqrtf(g / q->window_len) * sqrtf((float)(q->nfft)) );

  // scale window and copy
  for (i = 0; i < q->window_len; i++)
    q->w[i] = g * q->w[i];

  // reset the spgram object
  q->num_samples_total    = 0;
  q->num_transforms_total = 0;
  SPGRAM(_reset)(q);
  */
  // return new object
  return q;

}

CudaSpGramCF* CudaSpGramCF::create_default(unsigned int _nfft) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpGramCF::create_default(), fft size must be at least 2\n");
    exit(1);
  }

  return CudaSpGramCF::create(_nfft, LIQUID_WINDOW_KAISER, _nfft / 2, _nfft / 4);
}

void CudaSpGramCF::clear() {
  // clear FFT input
  unsigned int i;
// for (i = 0; i < _q->nfft; i++)
//   _q->buf_time[i] = 0.0f;

  // reset counters
  sample_timer   = delay;
  num_transforms = 0;
  num_samples    = 0;

  // clear PSD accumulation
//  for (i = 0; i < _q->nfft; i++)
  //  _q->psd[i] = 0.0f;
}

void CudaSpGramCF::reset() {

}


void CudaSpGramCF::push() {

}
void CudaSpGramCF::write() {

}
void CudaSpGramCF::step() {

}
void CudaSpGramCF::get_psd() {

}
void CudaSpGramCF::export_gnuplot( const char* _filename) {

}
void CudaSpGramCF::estimate_psd() {

}
