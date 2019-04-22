#ifndef HELIADES_FILM_H
#define HELIADES_FILM_H

#include <hermes/hermes.h>
#include <memory>

namespace heliades {

class Film {
public:
  struct Pixel {
    float xyz[3] = {0, 0, 0};
  };
  ///
  /// \param resolution
  Film(hermes::cuda::vec2u resolution);
  ~Film();
  Pixel *pixelsDeviceData();
  const Pixel *pixelsDeviceData() const;

  const hermes::cuda::vec2u fullResolution;

private:
  std::unique_ptr<Pixel[]> h_pixels; //!< host memory for pixels
  Pixel *d_pixels;                   //!< device memory for pixels
};

} // namespace heliades

#endif