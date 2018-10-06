// Copyright [2018] <Alexander Hurd>

#include <cuda-spgram-cf.h>
#include <liquid/liquid.h>
#include <chrono>
#include <iostream>

#define CUDA_OUTPUT_FILENAME "cuda_spgramcf_example.gnu"
#define LIQUID_OUTPUT_FILENAME "liquid_spgramcf_example.gnu"

int main() {

  // spectral periodogram options
  unsigned int nfft        =  	1<<20;  // spectral periodogram FFT size
  unsigned int num_samples =      100e6;  // number of samples
  unsigned int i;
  float psd[nfft];

  // generate signal
  std::cout << "generating input signal size : " << num_samples << std::endl;
  // generate signal
  //std::complex<float> y[num_samples];
  std::vector<std::complex<float>> *y = new std::vector<std::complex<float>>(num_samples);
  for (i = 0; i < num_samples; i++) {
    std::complex<float> s = 0;
    double theta = (double) i / (double) num_samples * M_PI;
    s.real(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));
    s.imag(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));

    // save
    (*y)[i] = s;
  }

  // start timer for cuda
  //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  //CudaSpGramCF::estimate_psd(nfft, y->data(), num_samples, psd);

  // create cuda spectral periodogram
  //CudaSpGramCF* q = CudaSpGramCF::create_default(nfft);
  CudaSpGramCF* q = CudaSpGramCF::create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft);
  q->print();
  //  print duration
  //std::cout << "cuda-spgram init duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

  std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

  q->write(y->data(), num_samples);
  q->get_psd(psd);

  //  print duration
  std::cout << "cuda-spgram write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t3).count() << std::endl;

  // start timer liquid


  //spgramcf_estimate_psd(nfft, y->data(), num_samples, psd);
  // create liquid spectral periodogram
   //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

  //spgramcf qq = spgramcf_create_default(nfft);
  spgramcf qq = spgramcf_create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft);
  spgramcf_print(qq);
  std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

  spgramcf_write(qq, y->data(), num_samples);
  spgramcf_get_psd(qq, psd);

  //  print duration
  std::cout << "liquid spgram write duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t4).count() << std::endl;


  //  print statistics
  printf("cuda spgram stats:\n");
  printf("total_num_samples : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_samples_total : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_transforms : %" PRIu64 "\n", q->get_num_transforms());
  printf("total_num_transforms_total : %" PRIu64 "\n", q->get_num_transforms_total());

  //  print statistics
  printf("liquid spgram stats:\n");
  printf("total_num_samples : %" PRIu64 "\n", spgramcf_get_num_samples(qq));
  printf("total_num_samples_total : %" PRIu64 "\n", spgramcf_get_num_samples_total(qq));
  printf("total_num_transforms : %" PRIu64 "\n", spgramcf_get_num_transforms(qq));
  printf("total_num_transforms_total : %" PRIu64 "\n", spgramcf_get_num_transforms_total(qq));

  // destroy object
  delete(y);
  delete(q);
  spgramcf_destroy(qq);

  printf("done.\n");
  return 0;
}
