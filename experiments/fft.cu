#include <cuda_runtime.h>
#include <cufft.h>
#include <hermes/hermes.h>
#include <iostream>

using namespace hermes::cuda;

#define NX 256
#define NY 128

__global__ void NormalizeIFFT(float *g_data, int width, int height, float N) {

  // index = x * height + y

  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  int index = yIndex * width + xIndex;

  g_data[index] = g_data[index] / N;
}

int main() {
  cufftReal input[NX][NY];
  for (int x = 0; x < NX; x++)
    for (int y = 0; y < NY; y++)
      input[x][y] = x * y;
  cufftReal *d_input;
  CUDA_CHECK(cudaMalloc((void **)&d_input, sizeof(cufftReal) * NX * NY));
  CUDA_CHECK(cudaMemcpy(d_input, input, sizeof(cufftReal) * NX * NY,
                        cudaMemcpyHostToDevice));
  cufftComplex *d_output;
  CUDA_CHECK(
      cudaMalloc((void **)&d_output, sizeof(cufftComplex) * NX * (NY / 2 + 1)));

  cufftHandle forwardPlan, inversePlan;
  if (cufftPlan2d(&forwardPlan, NX, NY, CUFFT_R2C) != CUFFT_SUCCESS) {
    std::cerr << "CUFFT Error: Failed to create plan\n";
    return -1;
  }
  if (cufftPlan2d(&inversePlan, NX, NY, CUFFT_C2R) != CUFFT_SUCCESS) {
    std::cerr << "CUFFT Error: Failed to create plan\n";
    return -1;
  }

  if (cufftExecR2C(forwardPlan, d_input, d_output) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
    return -1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (cufftExecC2R(inversePlan, d_output, d_input) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
    return -1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 grid(NX / 16, NY / 16, 1);
  dim3 threads(16, 16, 1);
  NormalizeIFFT<<<grid, threads>>>(d_input, NX, NY, NX * NY);

  CUDA_CHECK(cudaMemcpy(input, d_input, sizeof(cufftReal) * NX * NY,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int x = 0; x < NX; x++)
    for (int y = 0; y < NY; y++)
      std::cerr << input[x][y] << " == " << x * y << std::endl;

  cufftComplex output[NX * (NY / 2 + 1)];
  CUDA_CHECK(cudaMemcpy(output, d_output,
                        sizeof(cufftComplex) * NX * (NY / 2 + 1),
                        cudaMemcpyDeviceToHost));

  //   for (int x = 0; x < NX; x++)
  //     std::cerr << output[x].x << " " << output[x].y << std::endl;

  cufftDestroy(forwardPlan);
  cufftDestroy(inversePlan);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}