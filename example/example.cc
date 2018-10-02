// Copyright [2018] <Alexander Hurd>

#include <cuda-spgram-cf.h>
#include <chrono>
#include <iostream>

#define OUTPUT_FILENAME "cuda_spgramcf_example.gnu"

int main() {

  // spectral periodogram options
  unsigned int nfft        =  	1<<16; // spectral periodogram FFT size
  unsigned int num_samples =      1e6;  // number of samples
  unsigned int i;

  // create spectral periodogram
  CudaSpGramCF* q = CudaSpGramCF::create_default(nfft);
  //CudaSpGramCF* q = CudaSpGramCF::create(nfft, LIQUID_WINDOW_BLACKMANHARRIS7, nfft, nfft);
  q->print();

  // generate signal
  std::cout << "generating input signal size : " << num_samples << std::endl;
  // generate signal
  std::complex<float> y[num_samples];
  for (i = 0; i < num_samples; i++) {
    std::complex<float> s = 0;
    double theta = (double) i / (double) num_samples * M_PI;
    s.real(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));
    s.imag(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));

    // save
    y[i] = s;
  }

  // start timer
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  // push signal through periodogram
  std::cout << "writing samples to periodgram" << std::endl;
  q->write(y, num_samples);

  //  print duration
  std::cout << "duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

  //  export as gnu plot
  q->export_gnuplot(OUTPUT_FILENAME);

  //  print statistics
  printf("total_num_samples : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_samples_total : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_transforms : %" PRIu64 "\n", q->get_num_transforms());
  printf("total_num_transforms_total : %" PRIu64 "\n", q->get_num_transforms_total());

  // destroy object
  delete(q);

  printf("done.\n");
  return 0;
}
