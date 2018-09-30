// Copyright [2018] <Alexander Hurd>"

#include "cuda-spgram-cf.h"

CudaSpGramCF* CudaSpGramCF::create(unsigned int _nfft, int _wtype, unsigned int _window_len, unsigned int _delay) {
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


  cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * _nfft);
  cufftComplex* h_fft = (cufftComplex*)malloc(sizeof(cufftComplex) * _nfft);

  cufftComplex* d_signal;
  checkCudaErrors(cudaMalloc((void**)&d_signal, _nfft * sizeof(cufftComplex)));


  // create FFT arrays, object
  q->buf_time.resize(q->nfft);
  q->buf_freq.resize(q->nfft);
  q->psd.resize(q->nfft);
  checkCudaErrors(cufftPlan1d(&q->fft, q->nfft, CUFFT_C2C, 1));

  // create buffer
  //q->buffer = WINDOW(_create)(q->window_len);

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

  // reset the spgram object
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
  w.clear();
  psd.clear();
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
  for (i = 0; i < nfft; i++)
    psd[i] = 0.0f;
}

void CudaSpGramCF::reset() {
  // reset spgram object except for the window buffer
  this->clear();

  // clear the window buffer
  //  WINDOW(_reset)(_q->buffer);
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
  return num_samples_total;
}

uint64_t CudaSpGramCF::get_num_transforms_total() {
  return num_transforms_total;
}

void CudaSpGramCF::push(cufftComplex x) {
  // if buffer is full we need to pop
  if(buffer.size() == window_len) {
    buffer.pop();
  }

  // push sample into internal window
  buffer.push(x);

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

void CudaSpGramCF::write(cufftComplex* _x, size_t _n) {

}

void CudaSpGramCF::step() {
	 unsigned int i;

  // read buffer, copy to FFT input (applying window)
	for(i=0; i<window_len; i++){
	//	cufftComplex t = buffer[i];
 // for(auto it = buffer.begin(); it != buffer.end(); ++it){
 //   std::cout << *it << "\n";


  }

/*
	    // read buffer, copy to FFT input (applying window)
	    // TODO: use SIMD extensions to speed this up
	    TI * rc;
	    //WINDOW(_read)(_q->buffer, &rc);
	    for (i=0; i<_q->window_len; i++)
	        _q->buf_time[i] = rc[i] * _q->w[i];

	    // execute fft on _q->buf_time and store result in _q->buf_freq
	   // FFT_EXECUTE(_q->fft);

	    // accumulate output
	    // TODO: vectorize this operation
	    for (i=0; i<_q->nfft; i++) {
	        T v = crealf( _q->buf_freq[i] * conjf(_q->buf_freq[i]) );
	        if (_q->num_transforms == 0)
	            _q->psd[i] = v;
	        else
	            _q->psd[i] = _q->gamma*_q->psd[i] + _q->alpha*v;
	    }
*/
num_transforms++;
num_transforms_total++;
}

void CudaSpGramCF::get_psd(cufftComplex* _X) {
  // compute magnitude in dB and run FFT shift
  unsigned int i;
  unsigned int nfft_2 = nfft / 2;
  float scale = accumulate ? -10 * log10f(num_transforms) : 0.0f;
  // TODO: adjust scale if infinite integration
  /*	    for (i=0; i<_q->nfft; i++) {
  	        unsigned int k = (i + nfft_2) % _q->nfft;
  	        _X[i] = 10*log10f(_q->psd[k]+1e-6f) + scale;
  	    }
  	    */
}

int CudaSpGramCF::export_gnuplot(const char* _filename) {
  FILE* fid = fopen(_filename, "w");
  if (fid == NULL) {
    fprintf(stderr, "error: CudaSpGramCF export_gnuplot(), could not open '%s' for writing\n", _filename);
    return -1;
  }
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
  if (sample_rate < 0) {
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
  /* T* psd = (T*) malloc(_q->nfft * sizeof(T));
   SPGRAM(_get_psd)(_q, psd);
   unsigned int i;
   for (i = 0; i < _nfft; i++)
     fprintf(fid, "  %12.8f %12.8f\n", (float)i / (float)(nfft) - 0.5f, (float)(psd[i]));
   free(psd);
   fprintf(fid, "e\n");
  */
  // close it up
  fclose(fid);

  return 0;
}

void CudaSpGramCF::estimate_psd(unsigned int _nfft, cufftComplex* _x, unsigned int _n, cufftComplex* _psd) {
  // create object
  CudaSpGramCF* q = CudaSpGramCF::create_default(_nfft);

  // run spectral estimate on entire sequence
  q->write(_x, _n);

  // get PSD estimate
  q->get_psd(_psd);

  // destroy object
  delete q;
}
