// Copyright [2018] <Alexander Hurd>"

#include "cuda-spgram-cf.h"

// cuda kernels
#include "cuda-spgram-cf.cuh"

// cuda runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>

CudaSpGramCF* CudaSpGramCF::create(unsigned int _nfft,
                                   int _wtype,
                                   unsigned int _window_len,
                                   unsigned int _delay) {
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

  // create input cuda buffers on GPU
  checkCudaErrors(cudaMalloc((void**)&q->d_buffer, sizeof(std::complex<float>) * q->window_len));
  checkCudaErrors(cudaMalloc((void**)&q->d_buf_time, sizeof(cufftComplex) * q->nfft));

  // create psd that hold accumulated fft results
  q->psd.resize(q->nfft);
  checkCudaErrors(cudaMalloc((void**)&q->d_psd, sizeof(float) * q->nfft));
  checkCudaErrors(cudaMalloc((void**)&q->d_psd_out, sizeof(float) * q->nfft));

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

  //  allocate and copy window to device
  checkCudaErrors(cudaMalloc((void**)&q->d_w, sizeof(float) * q->window_len));
  checkCudaErrors(cudaMemcpy(q->d_w, q->w.data(), sizeof(float) * q->window_len, cudaMemcpyHostToDevice));

  // allocate d_index and sequence of size nfft
  checkCudaErrors(cudaMalloc((void**)&q->d_index, sizeof(uint64_t) * q->nfft));
  thrust::device_ptr<uint64_t> d_index_ptr = thrust::device_pointer_cast(q->d_index);
  thrust::sequence(d_index_ptr, d_index_ptr + q->nfft);

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

  return CudaSpGramCF::create(_nfft, LIQUID_WINDOW_KAISER, _nfft / 2, _nfft / 4);
}

CudaSpGramCF::~CudaSpGramCF() {
  // free allocated memory on GPU
  checkCudaErrors(cudaFree(d_buffer));
  checkCudaErrors(cudaFree(d_buf_time));
  checkCudaErrors(cudaFree(d_w));
  checkCudaErrors(cudaFree(d_psd));
  checkCudaErrors(cudaFree(d_psd_out));

  w.clear();
  psd.clear();

  checkCudaErrors(cufftDestroy(fft));
}

void CudaSpGramCF::clear() {
  thrust::device_ptr<liquid_float_complex> d_buffer_ptr = thrust::device_pointer_cast(d_buffer);
  thrust::generate(d_buffer_ptr, d_buffer_ptr + window_len, clear_liquid_float_complex());

  thrust::device_ptr<cufftComplex> d_buf_time_ptr = thrust::device_pointer_cast(d_buf_time);
  thrust::generate(d_buf_time_ptr, d_buf_time_ptr + nfft, clear_cufftComplex());

  // reset counters
  sample_timer   = delay;
  num_transforms = 0;
  num_samples    = 0;

  // clear PSD accumulation
  for (size_t i = 0; i < nfft; i++)
    psd[i] = 0.0f;

  thrust::device_ptr<float> d_psd_ptr = thrust::device_pointer_cast(d_psd);
  thrust::generate(d_psd_ptr, d_psd_ptr + nfft, clear_float());
  thrust::device_ptr<float> d_psd_out_ptr = thrust::device_pointer_cast(d_psd_out);
  thrust::generate(d_psd_out_ptr, d_psd_out_ptr + nfft, clear_float());

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

void CudaSpGramCF::push(std::complex<float> _x) {
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

void CudaSpGramCF::write(std::complex<float>* _x,
                         size_t _n) {
  // TODO: be smarter about how to write and execute samples
  unsigned int i;
  for (i = 0; i < _n; i++)
    this->push(_x[i]);
}

void CudaSpGramCF::step() {
  unsigned int i;

  // read buffer
  liquid_float_complex* rc;
  windowcf_read(buffer, &rc);

  // copy windowcf buffer to gpu
  checkCudaErrors(cudaMemcpy(d_buffer, rc, sizeof(liquid_float_complex) * window_len, cudaMemcpyHostToDevice));

  //apply window in gpu
  thrust::device_ptr<liquid_float_complex> d_buffer_ptr = thrust::device_pointer_cast(d_buffer);
  thrust::device_ptr<float> d_w_ptr = thrust::device_pointer_cast(d_w);
  thrust::device_ptr<cufftComplex> d_buf_time_ptr = thrust::device_pointer_cast(d_buf_time);
  thrust::transform(d_buffer_ptr, d_buffer_ptr + window_len, d_w_ptr, d_buf_time_ptr, apply_window());

  // execute fft on d_buf_time and store inplace
  checkCudaErrors(cufftExecC2C(fft, (cufftComplex*)d_buf_time, (cufftComplex*)d_buf_time, CUFFT_FORWARD));

  //  accumulate output in gpu
  thrust::device_ptr<uint64_t> d_index_ptr = thrust::device_pointer_cast(d_index);
  if(num_transforms == 0) {
    first_psd first_functor(d_buf_time, d_psd);
    thrust::for_each(d_index_ptr, d_index_ptr + nfft, first_functor);
  } else {
    accumulate_psd accumulate_functor(d_buf_time, d_psd, alpha, gamma);
    thrust::for_each(d_index_ptr, d_index_ptr + nfft, accumulate_functor);
  }

  num_transforms++;
  num_transforms_total++;
}

void CudaSpGramCF::get_psd(float* _X) {
  // compute magnitude in dB and run FFT shift in GPU
  float scale = accumulate ? -10 * log10f(num_transforms) : 0.0f;
  thrust::device_ptr<uint64_t> d_index_ptr = thrust::device_pointer_cast(d_index);
  calc_power_and_shift calc_power_and_shift_functor(d_psd, d_psd_out, scale, nfft);
  thrust::for_each(d_index_ptr, d_index_ptr + nfft, calc_power_and_shift_functor);

  //copy from device to output
  checkCudaErrors(cudaMemcpy(_X, d_psd_out, sizeof(float) * nfft, cudaMemcpyDeviceToHost));
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
    //liquid_get_scale(frequency, &unit, &g);
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
                                std::complex<float>* _x,
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
