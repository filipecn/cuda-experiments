#ifndef HELIADES_CORE_CAMERA_H
#define HELIADES_CORE_CAMERA_H

#include <heliades/geometry/ray.h>

namespace heliades {

class camera {
public:
  __host__ __device__ camera() {
    origin = hermes::cuda::point3(0.0, 0.0, 0.0);
    lowerLeftCorner = hermes::cuda::point3(-2.0, -1.0, -1.0);
    horizontal = hermes::cuda::vec3(4.0, 0.0, 0.0);
    vertical = hermes::cuda::vec3(0.0, 2.0, 0.0);
  }
  __host__ __device__ ray getRay(float u, float v) const {
    return ray(origin, lowerLeftCorner + horizontal * u + vertical * v);
  }
  hermes::cuda::point3 origin, lowerLeftCorner;
  hermes::cuda::vec3 horizontal, vertical;
};

} // namespace heliades

#endif // HELIADES_CORE_CAMERA_H