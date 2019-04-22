#include "io.h"

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ int rgbToInt(float r, float g, float b, float a) {
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  a = clamp(a, 0.0f, 255.0f);
  return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

texture<float, cudaTextureType2D> densityTex2;

__global__ void __pixelsToTexture(const heliades::Film::Pixel *in, int w, int h,
                                  unsigned int *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * w + x;
  if (x < w && y < h)
    out[index] =
        rgbToInt(in[index].xyz[0], in[index].xyz[1], in[index].xyz[2], 255);
}

void pixelsToTexture(unsigned int w, unsigned int h,
                     const heliades::Film::Pixel *in, unsigned int *out) {
  auto td = hermes::ThreadArrayDistributionInfo(w, h);
  __pixelsToTexture<<<td.gridSize, td.blockSize>>>(in, w, h, out);
}