#ifndef HELIADES_GEOMETRY_RAY_H
#define HELIADES_GEOMETRY_RAY_H

#include <hermes/geometry/cuda_point.h>

namespace heliades {

class ray {
public:
  __host__ __device__ ray() {}
  /// \param origin
  /// \param direction
  /// \return __host__ ray
  __host__ __device__ ray(const hermes::cuda::point3 &origin,
                          const hermes::cuda::vec3 &direction)
      : o(origin), d(direction) {}
  ///
  __host__ __device__ hermes::cuda::point3 operator()(float t) const {
    return o + d * t;
  }

  hermes::cuda::point3 o; //!< origin
  hermes::cuda::vec3 d;   //!< direction
};

} // namespace heliades

#endif // HELIADES_RAY_H