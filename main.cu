/* Just some trivial tests.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2016-04-18
 */
#include <stdio.h>
#include <iostream>
#include <sstream>

void checkCudaError( char const * const file, const int line );
void cudaFinalize(void);

#define SANE (checkCudaError(__FILE__, __LINE__))

template<class E>
class CRTPBase {
  public:
#ifdef HAS_USER_DEFINED_CAST
    __host__ __device__ __inline__
    operator E const&() const {
      return static_cast<const E &>(*this);
    }
#endif
};

class Impl : public CRTPBase<Impl> {
  public:
    double x;

  public:
    __host__ __device__ __inline__
    Impl(double x)
      : x(x)
    { }
};

template<class E>
double get_x(const CRTPBase<E> &crtp_base) {
  const E & e = static_cast<const E&>(crtp_base);
  return e.x;
}


__global__
void crtp_on_device_kernel(double * ret) {
  // The next three line will be referred to as (1)
  Impl impl(1.0);
  double x = get_x(impl);

  *ret = x;   // prevent NVCC from eliminating the kernel altogether.
}

void crtp_on_host() {
  // Note, these three lines are one-to-one copy of (1)
  Impl impl(1.0);
  double x = get_x(impl);

  printf("HURRAY, for the host.\n");
}

void crtp_on_device() {
  double * foo = NULL;
  cudaMalloc(&foo, sizeof(double));                                        SANE;

  crtp_on_device_kernel<<<1, 1>>>(foo);                                    SANE;
  printf("HURRAY, for the device.\n");

  cudaFree(foo);                                                           SANE;
  cudaFinalize();
}

int main() {
  crtp_on_host();
  crtp_on_device();

  return 0;
}

// ------------------------------------------------------------------------------
// ---- Pretty printing ---------------------------------------------------------

/// Clean up CUDA
void cudaFinalize(void){
  checkCudaError("before exit", -1);

  std::cout << ".. No CUDA-errors detected.\n";
  cudaDeviceReset();
}

/// Checks whether a CUDA error has been raised.
void checkCudaError(char const * const file, const int line ){
  cudaThreadSynchronize();
  if( cudaPeekAtLastError() != cudaSuccess ){
    std::stringstream ss;
    ss << "!! Error: " << file << ": " << line << ": "
       << cudaGetErrorString(cudaPeekAtLastError())
       << " error no.: " << cudaPeekAtLastError();
    ss << "\n";

    std::cout << ss.str();
    exit(EXIT_FAILURE);
  }
}
