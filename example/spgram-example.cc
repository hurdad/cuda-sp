// Copyright [2018] <Alexander Hurd>

#include <cuda-spgram-cf.h>
#include <liquid/liquid.h>
#include <chrono>
#include <iostream>

#define CUDA_OUTPUT_FILENAME "cuda_spgramcf_example.gnu"
#define LIQUID_OUTPUT_FILENAME "liquid_spgramcf_example.gnu"

int main() {

  // spectral periodogram options
  unsigned int nfft        =  	  1 << 15; // spectral periodogram FFT size
  unsigned int num_samples =         10e6; // number of samples
  float psd[nfft];                         // output

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

  // create cuda spectral periodogram
  //CudaSpGramCF* q = CudaSpGramCF::create_default(nfft);
  //CudaSpGramCF* q = CudaSpGramCF::create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft, CudaSpGramCF::DEVICE_MAPPED);
  CudaSpGramCF* q = CudaSpGramCF::create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft, CudaSpGramCF::UNIFIED);
  q->print();

  // start timer for cuda
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  //  write IQ data to periodgram
  q->write(y->data(), num_samples);

  //  populate output
  q->get_psd(psd);

  //  print write duration
  std::cout << "cuda-spgram write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

  // export plot
  q->export_gnuplot(CUDA_OUTPUT_FILENAME);

  //  print statistics
  printf("cuda spgram stats:\n");
  printf("total_num_samples : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_samples_total : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_transforms : %" PRIu64 "\n", q->get_num_transforms());
  printf("total_num_transforms_total : %" PRIu64 "\n", q->get_num_transforms_total());

  // create liquid spectral periodogram
  //spgramcf qq = spgramcf_create_default(nfft);
  spgramcf qq = spgramcf_create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft);
  spgramcf_print(qq);

  // start timer liquid
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

  //  write IQ data to periodgram
  spgramcf_write(qq, y->data(), num_samples);

  //  populate output
  spgramcf_get_psd(qq, psd);

  //  print write duration
  std::cout << "liquid spgram write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t2).count() << std::endl;

  // export plot
  spgramcf_export_gnuplot(qq, LIQUID_OUTPUT_FILENAME);

  //  print statistics
  printf("liquid spgram stats:\n");
  printf("total_num_samples : %" PRIu64 "\n", spgramcf_get_num_samples(qq));
  printf("total_num_samples_total : %" PRIu64 "\n", spgramcf_get_num_samples_total(qq));
  printf("total_num_transforms : %" PRIu64 "\n", spgramcf_get_num_transforms(qq));
  printf("total_num_transforms_total : %" PRIu64 "\n", spgramcf_get_num_transforms_total(qq));

  // destroy objects
  msourcecf_destroy(gen);
  delete(y);
  delete(q);
  spgramcf_destroy(qq);

  printf("done.\n");
  return 0;
}
