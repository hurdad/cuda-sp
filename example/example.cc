
#include <cuda-spgram-cf.h>

#define OUTPUT_FILENAME "cuda_spgramcf_example.gnu"

int main() {

  // spectral periodogram options
  unsigned int nfft        =   1024;  // spectral periodogram FFT size
  unsigned int num_samples =    2e6;  // number of samples

  unsigned int i;

  // derived values
  //float nstd = powf(10.0f, noise_floor/20.0f);

  // create spectral periodogram
  CudaSpGramCF* q = CudaSpGramCF::create_default(nfft);
  q->print();

  // generate signal
  for (i = 0; i < num_samples; i++) {
    std::complex<float> y = 0;
	double theta = (double) i / (double) num_samples * M_PI;
	y.real(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));
	y.imag(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));

    // push resulting sample through periodogram
    q->push(y);
  }

  q->export_gnuplot(OUTPUT_FILENAME);

  // destroy object
  delete(q);

  printf("done.\n");
  return 0;

}
