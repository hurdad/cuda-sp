// Copyright [2018] <Alexander Hurd>

#include <cuda-spwaterfall-cf.h>
#include <liquid/liquid.h>
#include <chrono>
#include <iostream>

#define CUDA_OUTPUT_FILENAME "cuda_spwaterfallcf_example"
#define LIQUID_OUTPUT_FILENAME "liquid_spwaterfallcf_example"

int main() {

  // spectral periodogram options
  unsigned int nfft        =  	  1 << 16; // spectral periodogram FFT size
  unsigned int time        =  	  	  250; // minimum time buffer
  unsigned int num_samples =         10e6; // number of samples

  // generate QPSK signal
  std::cout << "generating QPSK signal size : " << num_samples << std::endl;
  std::vector<std::complex<float>>* y = new std::vector<std::complex<float>>(num_samples);

  // create stream generator
  msourcecf gen = msourcecf_create();

  // add noise source (narrow-band)
  int id_noise = msourcecf_add_noise(gen, 0.10f);
  msourcecf_set_frequency(gen, id_noise, 0.4 * 2 * M_PI);
  msourcecf_set_gain     (gen, id_noise, -20.0f);

  // add tone
  int id_tone = msourcecf_add_tone(gen);
  msourcecf_set_frequency(gen, id_tone, -0.4 * 2 * M_PI);
  msourcecf_set_gain     (gen, id_tone, -10.0f);

  // add modulated data
  int id_modem = msourcecf_add_modem(gen, LIQUID_MODEM_QPSK, 4, 12, 0.30f);
  msourcecf_set_frequency(gen, id_modem, -0.1 * 2 * M_PI);
  msourcecf_set_gain     (gen, id_modem, 0.0f);

  // generate samples into y
  msourcecf_write_samples(gen, y->data(), num_samples);

  // create cuda spectral waterfall
  //CudaSpWaterfallCF* q = CudaSpWaterfallCF::create_default(nfft, time);
  CudaSpWaterfallCF* q = CudaSpWaterfallCF::create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft, time, CudaSpGramCF::DEVICE_MAPPED);
  q->print();

  // start timer for cuda
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  //  write IQ data to periodgram
  q->write(y->data(), num_samples);

  //  print write duration
  std::cout << "cuda-spwaterfall write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

  // export plot
  q->set_rate(100e6);
  q->set_freq(750e6);
  q->set_dims(1200, 800);
  q->export_files(CUDA_OUTPUT_FILENAME);

  // create liquid spectral periodogram
  spwaterfallcf qq = spwaterfallcf_create_default(nfft, time);
  //spwaterfallcf qq = spwaterfallcf_create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft, time);
  spwaterfallcf_print(qq);

  // start timer liquid
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

  //  write IQ data to periodgram
  spwaterfallcf_write(qq, y->data(), num_samples);

  //  print write duration
  std::cout << "liquid spwaterfall write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t2).count() << std::endl;

  // export plot
  spwaterfallcf_set_rate(qq, 100e6);
  spwaterfallcf_set_freq(qq, 750e6);
  spwaterfallcf_set_dims(qq, 1200, 800);
  spwaterfallcf_export(qq, LIQUID_OUTPUT_FILENAME);

  // destroy objects
  msourcecf_destroy(gen);
  delete(y);
  delete(q);
  spwaterfallcf_destroy(qq);

  printf("done.\n");
  return 0;
}
