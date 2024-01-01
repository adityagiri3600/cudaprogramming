#include <iostream>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define CPU_LIMIT 1000000000
#define MAX_THREADS 1024

void listCudaDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads in X-dimension of block: " << deviceProp.maxThreadsDim[0] << std::endl;
        std::cout << "  Max threads in Y-dimension of block: " << deviceProp.maxThreadsDim[1] << std::endl;
        std::cout << "  Max threads in Z-dimension of block: " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Max blocks in X-dimension of grid: " << deviceProp.maxGridSize[0] << std::endl;
        std::cout << "  Max blocks in Y-dimension of grid: " << deviceProp.maxGridSize[1] << std::endl;
        std::cout << "  Max blocks in Z-dimension of grid: " << deviceProp.maxGridSize[2] << std::endl;
    }
}

__global__
void add_cuda(int *a, int *b, int *c, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void add_cpu(int *a, int *b, int *c, int n)
{
  if(n>=CPU_LIMIT) {
    return;
  }
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(void)
{
  listCudaDevices();
  for (int n = 1000000; n < 10000000000; n*=10) {
    
    int *a, *b, *c;
    int size = n * sizeof(int);

    a = new int[n];
    b = new int[n];
    c = new int[n];

    for (int i = 0; i < n; i++) {
      a[i] = i;
      b[i] = i;
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_cpu(a, b, c, n);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = stop_cpu - start_cpu;
    std::cout << "CPU elapsed time: " << duration_ms.count() << " ms" << std::endl;

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    add_cuda<<<ceil(n/(float)MAX_THREADS), MAX_THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();
    duration_ms = stop_gpu - start_gpu;
    std::cout << "GPU elapsed time: " << duration_ms.count() << " ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }

  return 0;
}