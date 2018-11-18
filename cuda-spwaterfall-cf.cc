// Copyright [2018] <Alexander Hurd>"

#include "cuda-spwaterfall-cf.h"

CudaSpWaterfallCF* CudaSpWaterfallCF::create(unsigned int _nfft,
    int          _wtype,
    unsigned int _window_len,
    unsigned int _delay,
    unsigned int _time,
    CudaSpGramCF::CudaMemoryAPI_t _api) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), fft size must be at least 2\n");
    exit(1);
  } else if (_window_len > _nfft) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), window size cannot exceed fft size\n");
    exit(1);
  } else if (_window_len == 0) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), window size must be greater than zero\n");
    exit(1);
  } else if (_wtype == LIQUID_WINDOW_KBD && _window_len % 2) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), KBD window length must be even\n");
    exit(1);
  } else if (_delay == 0) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), delay must be greater than 0\n");
    exit(1);
  } else if (_time == 0) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), time must be greater than 0\n");
    exit(1);
  } else if ((_api != CudaSpGramCF::DEVICE_MAPPED) && (_api != CudaSpGramCF::UNIFIED)) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create(), api must be valid\n");
    exit(1);
  }

// allocate memory for main object
  CudaSpWaterfallCF* q = new CudaSpWaterfallCF();

// set input parameters
  q->nfft         = _nfft;
  q->time         = _time;
  q->frequency    =  0;
  q->sample_rate  = -1;
  q->width        = 800;
  q->height       = 800;

// create buffer to hold aggregated power spectral density
// NOTE: the buffer is two-dimensional time/frequency grid that is two times
//       'nfft' and 'time' to account for log-average consolidation each time
//       the buffer gets filled
  q->psd.resize(2 * q->nfft * q->time);

// create spectral periodogram object
  q->periodogram = CudaSpGramCF::create(_nfft, _wtype, _window_len, _delay, _api);

// reset the object
  q->reset();

// return new object
  return q;
}

CudaSpWaterfallCF* CudaSpWaterfallCF::create_default(unsigned int _nfft, unsigned int _time) {
  // validate input
  if (_nfft < 2) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create_default(), fft size must be at least 2\n");
    exit(1);
  } else if (_time < 2) {
    fprintf(stderr, "error: CudaSpWaterfallCF::create_default(), fft size must be at least 2\n");
    exit(1);
  }

  return CudaSpWaterfallCF::create(_nfft, LIQUID_WINDOW_KAISER, _nfft / 2, _nfft / 4, _time, CudaSpGramCF::DEVICE_MAPPED);
}

CudaSpWaterfallCF::~CudaSpWaterfallCF() {
  // free allocated memory
  psd.clear();

  // destroy internal spectral periodogram object
  delete periodogram;
}

void CudaSpWaterfallCF::clear() {
  memset(psd.data(), 0x00, 2 * nfft * time * sizeof(float));
  index_time = 0;
}

void CudaSpWaterfallCF::reset() {
  this->clear();
  rollover = 1;
}

void CudaSpWaterfallCF::print() {
  printf("CudaSpWaterfallCF: nfft=%u, time=%u\n", nfft, time);
}

int CudaSpWaterfallCF::set_freq(float _freq) {
  frequency = _freq;
  return 0;
}

int CudaSpWaterfallCF::set_rate(float _rate) {
  // validate input
  if (_rate <= 0.0f) {
    fprintf(stderr, "error: CudaSpWaterfallCF::set_rate(), sample rate must be greater than zero\n");
    return -1;
  }
  sample_rate = _rate;
  return 0;
}

int CudaSpWaterfallCF::set_dims(unsigned int _width,
                                unsigned int _height) {
  width  = _width;
  height = _height;
  return 0;
}

void CudaSpWaterfallCF::push(liquid_float_complex _x) {
  periodogram->push(_x);
  this->step();
}

void CudaSpWaterfallCF::write(liquid_float_complex* _x,
                              size_t _n) {
  // TODO: be smarter about how to write and execute samples
  unsigned int i;
  for (i = 0; i < _n; i++)
    this->push(_x[i]);
}

int CudaSpWaterfallCF::export_files(const char* _base) {
  return export_bin(_base) + export_gnu(_base);
}

void CudaSpWaterfallCF::step() {
  // determine if we need to extract PSD estimate from periodogram
  if (periodogram->get_num_transforms() >= rollover) {
    // get PSD estimate from periodogram object, placing result in
    // proper location in internal buffer
    periodogram->get_psd(psd.data() + nfft * index_time);

    // soft reset of internal state, counters
    periodogram->clear();

    // increment buffer counter
    index_time++;

    // determine if buffer is full and we need to consolidate buffer
    if (index_time == 2 * time)
      this->consolidate_buffer();
  }
}

void CudaSpWaterfallCF::consolidate_buffer() {
  // assert(_q->index_time == 2*_q->time);
  printf("consolidating... (rollover = %10u, total samples : %16llu, index : %u)\n",
         rollover, periodogram->get_num_samples_total(), index_time);
  unsigned int i; // time index
  unsigned int k; // freq index
  for (i = 0; i < time; i++) {
    for (k = 0; k < nfft; k++) {
      // compute median
      float v0 = psd[ (2 * i + 0) * nfft + k ];
      float v1  = psd[ (2 * i + 1) * nfft + k ];

      // keep log average (only need double buffer for this, not triple buffer)
      psd[ i * nfft + k ] = logf(0.5f * (expf(v0) + expf(v1)));
    }
  }

  // update time index
  index_time = time;

  // update rollover counter
  rollover *= 2;
}

int CudaSpWaterfallCF::export_bin(const char* _base) {
  // add '.bin' extension to base
  int n = strlen(_base);
  char filename[n + 5];
  sprintf(filename, "%s.bin", _base);

  // open output file for writing
  FILE* fid = fopen(filename, "w");
  if (fid == NULL) {
    fprintf(stderr, "error: CudaSpWaterfallCF::export_bin(), could not open '%s' for writing\n",
            filename);
    return -1;
  }

  unsigned int i;

  // write header
  float nfftf = (float)(nfft);
  fwrite(&nfftf, sizeof(float), 1, fid);
  for (i = 0; i < nfft; i++) {
    float f = (float)i / nfftf - 0.5f;
    fwrite(&f, sizeof(float), 1, fid);
  }

  // write output spectral estimate
  // TODO: force converstion from type 'T' to type 'float'
  uint64_t total_samples = periodogram->get_num_samples_total();
  for (i = 0; i < index_time; i++) {
    float n = (float)i / (float)(index_time) * (float)total_samples;
    fwrite(&n, sizeof(float), 1, fid);
    fwrite(&psd[i * nfft], sizeof(float), nfft, fid);
  }

  // close it up
  fclose(fid);
  printf("results written to %s\n", filename);
  return 0;
}

int CudaSpWaterfallCF::export_gnu(const char* _base) {
  // add '.bin' extension to base
  int n = strlen(_base);
  char filename[n + 5];
  sprintf(filename, "%s.gnu", _base);

  // open output file for writing
  FILE* fid = fopen(filename, "w");
  if (fid == NULL) {
    fprintf(stderr, "error: CudaSpWaterfallCF::export_gnu(), could not open '%s' for writing\n",
            filename);
    return -1;
  }

  // scale to thousands, millions, billions (etc.) automatically
  uint64_t total_samples = periodogram->get_num_samples_total();
  char units  = ' ';
  float scale = 1.0f;
  liquid_get_scale((float)total_samples / 4, &units, &scale);

  fprintf(fid, "#!/usr/bin/gnuplot\n");
  fprintf(fid, "reset\n");
  fprintf(fid, "set terminal png size %u,%u enhanced font 'Verdana,10'\n", width, height);
  fprintf(fid, "set output '%s.png'\n", _base);
  fprintf(fid, "unset key\n");
  fprintf(fid, "set style line 11 lc rgb '#808080' lt 1\n");
  fprintf(fid, "set border 3 front ls 11\n");
  fprintf(fid, "set style line 12 lc rgb '#888888' lt 0 lw 1\n");
  fprintf(fid, "set grid front ls 12\n");
  fprintf(fid, "set tics nomirror out scale 0.75\n");
  fprintf(fid, "set yrange [0:%f]\n", (float)(total_samples - 1)*scale);
  fprintf(fid, "set ylabel 'Sample Index'\n");
  fprintf(fid, "set format y '%%.0f %c'\n", units);
  fprintf(fid, "# disable colorbar tics\n");
  fprintf(fid, "set cbtics scale 0\n");
  fprintf(fid, "set palette negative defined ( \\\n");
  fprintf(fid, "    0 '#D53E4F',\\\n");
  fprintf(fid, "    1 '#F46D43',\\\n");
  fprintf(fid, "    2 '#FDAE61',\\\n");
  fprintf(fid, "    3 '#FEE08B',\\\n");
  fprintf(fid, "    4 '#E6F598',\\\n");
  fprintf(fid, "    5 '#ABDDA4',\\\n");
  fprintf(fid, "    6 '#66C2A5',\\\n");
  fprintf(fid, "    7 '#3288BD' )\n");
  fprintf(fid, "\n");
  if (sample_rate < 0) {
    fprintf(fid, "set xrange [-0.5:0.5]\n");
    float xtics = 0.1f;
    fprintf(fid, "set xtics %f\n", xtics);
    fprintf(fid, "set xlabel 'Normalized Frequency [f/F_s]'\n");
    fprintf(fid, "plot '%s.bin' u 1:($2*%e):3 binary matrix with image\n", _base, scale);
  } else {
    char unit;
    float g = 1.0f;
    float f_hi = frequency + 0.5f * sample_rate; // highest frequency
    liquid_get_scale(f_hi / 2, &unit, &g);
    fprintf(fid, "set xlabel 'Frequency [%cHz]'\n", unit);
    // target xtics spacing roughly every 60-80 pixels
    float xn = ((float) width * 0.8f) / 70.0f;  // rough number of tics
    //float xs = _q->sample_rate * g / xn;            // normalized spacing
    float xt = 1.0f;                                // round to nearest 1, 2, 5, or 10
    // potential xtic spacings
    float spacing[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, -1.0f};
    unsigned int i = 0;
    while (spacing[i] > 0) {
      if (sample_rate * g / spacing[i] < 1.2f * xn) {
        xt = spacing[i];
        break;
      }
      i++;
    }
    //printf("xn:%f, xs:%f, xt:%f\n", xn, xs, xt);
    fprintf(fid, "set xrange [%f:%f]\n", g * (frequency - 0.5 * sample_rate), g * (frequency + 0.5 * sample_rate));
    fprintf(fid, "set xtics %f\n", xt);
    fprintf(fid, "plot '%s.bin' u ($1*%f+%f):($2*%e):3 binary matrix with image\n",
            _base,
            g * (sample_rate < 0 ? 1 : sample_rate),
            g * frequency,
            scale);
  }
  fclose(fid);

  // close it up
  printf("results written to %s\n", filename);
  printf("index time       : %u\n", index_time);
  printf("rollover         : %u\n", rollover);
  printf("total transforms : %llu\n", periodogram->get_num_transforms_total());
  return 0;
}
