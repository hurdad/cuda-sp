#include <benchmark/benchmark.h>

#include <vector>
#include <string>
#include <complex>
#include <liquid/liquid.h>

#include "cuda-spgram-cf.h"

#define DATA_MULTIPLIER (10)

void generate_signal(std::complex<float>* signal, const int N) {
  int i;
  for (i = 0; i < N; ++i) {
    double theta = (double) i / (double) N * M_PI;
    signal[i].real(1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta));
    signal[i].imag(1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta));
  }
}

static void cuda_spgram(benchmark::State& state) {
  int N = state.range(0);

  CudaSpGramCF* q = CudaSpGramCF::create(N, LIQUID_WINDOW_BLACKMANHARRIS7, N, N, CudaSpGramCF::DEVICE_MAPPED);

  std::vector<std::complex<float>> signal(N * DATA_MULTIPLIER);
  generate_signal(signal.data(), N * DATA_MULTIPLIER);

  for (auto _ : state) {
    //  write IQ data to periodgram
    q->write(signal.data(), N * DATA_MULTIPLIER);
  }

  delete(q);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * DATA_MULTIPLIER);
  state.SetBytesProcessed(
    static_cast<int64_t>(state.iterations()) * N * DATA_MULTIPLIER * sizeof(std::complex<float>));
  state.SetComplexityN(N);
}
BENCHMARK(cuda_spgram)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Complexity();

static void liquid_spgram(benchmark::State& state) {
  int N = state.range(0);

  spgramcf qq = spgramcf_create(N, LIQUID_WINDOW_BLACKMANHARRIS7, N, N);

  std::vector<std::complex<float>> signal(N * DATA_MULTIPLIER);
  generate_signal(signal.data(), N * DATA_MULTIPLIER);

  for (auto _ : state) {
    //  write IQ data to periodgram
    spgramcf_write(qq, signal.data(), N * DATA_MULTIPLIER);
  }

  spgramcf_destroy(qq);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * DATA_MULTIPLIER);
  state.SetBytesProcessed(
    static_cast<int64_t>(state.iterations()) * N * DATA_MULTIPLIER * sizeof(std::complex<float>));
  state.SetComplexityN(N);
}
BENCHMARK(liquid_spgram)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Complexity();
BENCHMARK_MAIN();
