// Copyright [2018] <Alexander Hurd>"

#include "cuda-spgram-cf.h"

// cuda runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>

CudaSpGramCF* CudaSpGramCF::create(unsigned int _nfft,
                                   int _wtype,
                                   unsigned int _window_len,
                                   unsigned int _delay,
                                   CudaMemoryAPI_t _api) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpGramCF::create(), fft size must be at least 2\n");
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
  } else if ((_api != CudaSpGramCF::DEVICE_MAPPED) && (_api != CudaSpGramCF::UNIFIED)) {
    fprintf(stderr, "error: CudaSpGramCF::create(), api must be valid\n");
    exit(1);
  }

  // allocate memory for main object
  CudaSpGramCF* q = new CudaSpGramCF();

  // set input parameters
  q->nfft       = _nfft;
  q->wtype      = _wtype;
  q->window_len = _window_len;
  q->delay      = _delay;
  q->api 		= _api;
  q->frequency  =  0;
  q->sample_rate = -1;

  // set object for full accumulation
  q->set_alpha(-1.0f);

  // create cuda FFT arrays, object
  if(q->api == DEVICE_MAPPED) {
    q->buf_time = (cufftComplex*)malloc(sizeof(cufftComplex) * q->nfft);
    q->buf_freq = (cufftComplex*)malloc(sizeof(cufftComplex) * q->nfft);
    checkCudaErrors(cudaMalloc((void**)&q->d_buf_time, sizeof(cufftComplex) * q->nfft));
  }
  if(q->api == UNIFIED) {
    checkCudaErrors(cudaMallocManaged(&q->buf_time, sizeof(cufftComplex) * q->nfft));
    checkCudaErrors(cudaMallocManaged(&q->buf_freq, sizeof(cufftComplex) * q->nfft));
  }
  q->psd.resize(q->nfft);

  // init plan
  checkCudaErrors(cufftPlan1d(&q->fft, q->nfft, CUFFT_C2C, 1));

  // create buffer
  q->buffer = windowcf_create(q->window_len);

  // create window
  q->w.resize(q->window_len);
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
      fprintf(stderr, "error: CudaSpGramCF::create(), invalid window\n");
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

  // reset the object
  q->num_samples_total    = 0;
  q->num_transforms_total = 0;
  q->reset();

  // return new object
  return q;
}

CudaSpGramCF* CudaSpGramCF::create_default(unsigned int _nfft) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpGramCF::create_default(), fft size must be at least 2\n");
    exit(1);
  }

  return CudaSpGramCF::create(_nfft, LIQUID_WINDOW_KAISER, _nfft / 2, _nfft / 4, DEVICE_MAPPED);
}

CudaSpGramCF::~CudaSpGramCF() {
  // free allocated memory
  if(api == DEVICE_MAPPED) {
    free(buf_time);
    free(buf_freq);
    checkCudaErrors(cudaFree(d_buf_time));
  }

  if(api == UNIFIED) {
    checkCudaErrors(cudaFree(buf_time));
    checkCudaErrors(cudaFree(buf_freq));
  }

  w.clear();
  psd.clear();

  //fftwf_destroy_plan(fft);
  checkCudaErrors(cufftDestroy(fft));
}

void CudaSpGramCF::clear() {
  // clear FFT input
  unsigned int i;
  for (i = 0; i < nfft; i++) {
    buf_time[i].x = 0.0f;
    buf_time[i].y = 0.0f;
  }

  // reset counters
  sample_timer   = delay;
  num_transforms = 0;
  num_samples    = 0;

  // clear PSD accumulation
  for (i = 0; i < nfft; i++)
    psd[i] = 0.0f;
}

void CudaSpGramCF::reset() {
  // reset spgram object except for the window buffer
  this->clear();

  // clear the window buffer
  windowcf_reset(buffer);
}

void CudaSpGramCF::print() {
  printf("CudaSpGramCF: nfft=%u, window=%u, delay=%u\n", nfft, window_len, delay);
}

// set forgetting factor
int CudaSpGramCF::set_alpha(float _alpha) {
  // validate input
  if (_alpha != -1 && (_alpha < 0.0f || _alpha > 1.0f)) {
    fprintf(stderr, "warning: CudaSpGramCF set_alpha(), alpha must be in {-1,[0,1]}\n");
    return -1;
  }

  // set accumulation flag appropriately
  accumulate = (_alpha == -1.0f) ? 1 : 0;

  if (accumulate) {
    alpha = 1.0f;
    gamma = 1.0f;
  } else {
    alpha = _alpha;
    gamma = 1.0f - alpha;
  }
  return 0;
}

void CudaSpGramCF::set_freq(float _freq) {
  frequency = _freq;
}

int CudaSpGramCF::set_rate(float _rate) {
  // validate input
  if (_rate <= 0.0f) {
    fprintf(stderr, "error: CudaSpGramCF set_rate(), sample rate must be greater than zero\n");
    return -1;
  }
  sample_rate = _rate;
  return 0;
}

size_t CudaSpGramCF::get_nfft() {
  return nfft;
}

size_t CudaSpGramCF::get_window_len() {
  return window_len;
}

size_t CudaSpGramCF::get_delay() {
  return delay;
}

uint64_t CudaSpGramCF::get_num_samples() {
  return num_samples;
}

uint64_t CudaSpGramCF::get_num_samples_total() {
  return num_samples_total;
}

uint64_t CudaSpGramCF::get_num_transforms() {
  return num_transforms;
}

uint64_t CudaSpGramCF::get_num_transforms_total() {
  return num_transforms_total;
}

void CudaSpGramCF::push(liquid_float_complex _x) {
  // push sample into internal window
  windowcf_push(buffer, _x);

  // update counters
  num_samples++;
  num_samples_total++;

  // adjust timer
  sample_timer--;

  if(sample_timer)
    return;

  // reset timer and step through computation
  sample_timer = delay;
  this->step();
}

void CudaSpGramCF::write(liquid_float_complex* _x,
                         size_t _n) {
  // TODO: be smarter about how to write and execute samples
  unsigned int i;
  for (i = 0; i < _n; i++)
    this->push(_x[i]);
}

void CudaSpGramCF::step() {
  unsigned int i;

  // read buffer, copy to FFT input (applying window)
  std::complex<float>* rc;
  windowcf_read(buffer, &rc);
  for (i = 0; i < window_len; i++) {
    buf_time[i].x = rc[i].real() * w[i];
    buf_time[i].y = rc[i].imag() * w[i];
  }

  if(api == DEVICE_MAPPED) {
    //  copy host buff_time to device
    checkCudaErrors(cudaMemcpy(d_buf_time, buf_time, sizeof(cufftComplex)* nfft, cudaMemcpyHostToDevice));

    // execute fft on dev_buf_time and store inplace
    checkCudaErrors(cufftExecC2C(fft, (cufftComplex*)d_buf_time, (cufftComplex*)d_buf_time, CUFFT_FORWARD));

    // Copy device dev_buf_time to host buf_freq
    checkCudaErrors(cudaMemcpy(buf_freq, d_buf_time, sizeof(cufftComplex)* nfft, cudaMemcpyDeviceToHost));
  }

  if(api == UNIFIED) {
    // execute fft on buf_time and store in buf_freq
    checkCudaErrors(cufftExecC2C(fft, (cufftComplex*)buf_time, (cufftComplex*)buf_freq, CUFFT_FORWARD));
    cudaDeviceSynchronize();
  }

  // accumulate output
  // TODO: vectorize this operation
  for (i = 0; i < nfft; i++) {
    liquid_float_complex freq((float)buf_freq[i].x, (float)buf_freq[i].y);
    liquid_float_complex confj((float)buf_freq[i].x, (float)buf_freq[i].y * -1);
    liquid_float_complex t = freq * confj;
    float v = t.real();
    if (num_transforms == 0)
      psd[i] = v;
    else
      psd[i] = gamma * psd[i] + alpha * v;
  }

  num_transforms++;
  num_transforms_total++;
}

void CudaSpGramCF::get_psd(float* _X) {
  // compute magnitude in dB and run FFT shift
  unsigned int i;
  unsigned int nfft_2 = nfft / 2;
  float scale = accumulate ? -10 * log10f(num_transforms) : 0.0f;
  // TODO: adjust scale if infinite integration
  for (i = 0; i < nfft; i++) {
    unsigned int k = (i + nfft_2) % nfft;
    _X[i] = 10 * log10f(psd[k] + 1e-6f) + scale;
  }
}

int CudaSpGramCF::export_gnuplot(const char* _filename) {
  FILE* fid = fopen(_filename, "w");
  if (fid == NULL) {
    fprintf(stderr, "error: CudaSpGramCF export_gnuplot(), could not open '%s' for writing\n", _filename);
    return -1;
  }
  fprintf(fid, "#!/usr/bin/gnuplot\n");
  fprintf(fid, "# %s : auto-generated file\n", _filename);
  fprintf(fid, "reset\n");
  fprintf(fid, "set terminal png size 1200,800 enhanced font 'Verdana,10'\n");
  fprintf(fid, "set output '%s.png'\n", _filename);
  fprintf(fid, "set autoscale y\n");
  fprintf(fid, "set ylabel 'Power Spectral Density'\n");
  fprintf(fid, "set style line 12 lc rgb '#404040' lt 0 lw 1\n");
  fprintf(fid, "set grid xtics ytics\n");
  fprintf(fid, "set grid front ls 12\n");
  //fprintf(fid,"set style fill transparent solid 0.2\n");
  const char plot_with[] = "lines"; // "filledcurves x1"
  fprintf(fid, "set nokey\n");
  if(sample_rate < 0) {
    fprintf(fid, "set xrange [-0.5:0.5]\n");
    fprintf(fid, "set xlabel 'Noramlized Frequency'\n");
    fprintf(fid, "plot '-' w %s lt 1 lw 2 lc rgb '#004080'\n", plot_with);
  } else {
    char unit = ' ';
    float g = 1.0f;
    liquid_get_scale(frequency, &unit, &g);
    fprintf(fid, "set xlabel 'Frequency [%cHz]'\n", unit);
    fprintf(fid, "set xrange [%f:%f]\n", g * (frequency - 0.5 * sample_rate), g * (frequency + 0.5 * sample_rate));
    fprintf(fid, "plot '-' u ($1*%f+%f):2 w %s lt 1 lw 2 lc rgb '#004080'\n",
            g * (sample_rate < 0 ? 1 : sample_rate), g * frequency, plot_with);
  }

  // export spectrum data
  float psd[nfft];
  get_psd(psd);
  unsigned int i;
  for (i = 0; i < nfft; i++)
    fprintf(fid, "  %12.8f %12.8f\n", (float)i / (float)(nfft) - 0.5f, (float)(psd[i]));

  fprintf(fid, "end\n");

  // close it up
  fclose(fid);

  return 0;
}

void CudaSpGramCF::estimate_psd(unsigned int _nfft,
                                liquid_float_complex* _x,
                                unsigned int _n,
                                float* _psd) {
  // create object
  CudaSpGramCF* q = CudaSpGramCF::create_default(_nfft);

  // run spectral estimate on entire sequence
  q->write(_x, _n);

  // get PSD estimate
  q->get_psd(_psd);

  // destroy object
  delete q;
}
