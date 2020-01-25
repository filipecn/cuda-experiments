#include "film.h"

namespace heliades {

Film::Film(hermes::cuda::vec2u resolution) : fullResolution(resolution) {
  // allocate data
  using namespace hermes::cuda;
  CUDA_CHECK(cudaMalloc((void **)&d_pixels,
                        sizeof(Pixel) * fullResolution.x * fullResolution.y));
  h_pixels =
      std::unique_ptr<Pixel[]>(new Pixel[fullResolution.x * fullResolution.y]);
}

Film::~Film() {
  if (d_pixels)
    cudaFree(d_pixels);
}

Film::Pixel *Film::pixelsDeviceData() { return d_pixels; }

const Film::Pixel *Film::pixelsDeviceData() const { return d_pixels; }

} // namespace heliades