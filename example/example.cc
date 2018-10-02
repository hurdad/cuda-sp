// Copyright [2018] <Alexander Hurd>

#include <cuda-spgram-cf.h>
#include <chrono>
#include <iostream>

#define OUTPUT_FILENAME "cuda_spgramcf_example.gnu"

int main() {

  // spectral periodogram options
  unsigned int nfft        =   131072;  // spectral periodogram FFT size
  unsigned int num_samples =      1e6;  // number of samples

  unsigned int i;

  // create spectral periodogram
  CudaSpGramCF* q = CudaSpGramCF::create_default(nfft);
  q->print();

  // start timer
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  // generate signal
  for (i = 0; i < num_samples; i++) {
    std::complex<float> y = 0;
    double theta = (double) i / (double) num_samples * M_PI;
    y.real(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));
    y.imag(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));

    // push resulting sample through periodogram
    q->push(y);
  }

  //  export as gnu plot
  q->export_gnuplot(OUTPUT_FILENAME);

  //  show statistics
  std::cout << "duration ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;
  printf("total_num_samples : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_samples_total : %" PRIu64 "\n", q->get_num_samples());
  printf("total_num_transforms : %" PRIu64 "\n", q->get_num_transforms());
  printf("total_num_transforms_total : %" PRIu64 "\n", q->get_num_transforms_total());

  // destroy object
  delete(q);

  printf("done.\n");
  return 0;
}
