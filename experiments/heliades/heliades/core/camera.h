#ifndef HELIADES_CORE_CAMERA_H
#define HELIADES_CORE_CAMERA_H

#include <heliades/common/random.h>
#include <heliades/geometry/ray.h>

namespace heliades {

inline __host__ __device__ hermes::cuda::vec3 randomInUnitDisk(HS &rng) {
  hermes::cuda::vec3 p;
  do {
    p = 2.f * hermes::cuda::vec3(rng.random(), rng.random(), 0) -
        hermes::cuda::vec3(1, 1, 0);
  } while (p.length2() >= 1);
  return p;
}

class camera {
public:
  __host__ __device__ camera(hermes::cuda::point3 lookFrom,
                             hermes::cuda::point3 lookAt,
                             hermes::cuda::vec3 vup, float vfov, float aspect,
                             float aperture, float focusDist) {
    lensRadius = aperture / 2;
    float theta = vfov * 3.14159265358979323846f / 180;
    float halfHeight = tanf(theta / 2);
    float halfWidth = aspect * halfHeight;
    origin = lookFrom;
    w = hermes::cuda::normalize(lookFrom - lookAt);
    u = hermes::cuda::normalize(hermes::cuda::cross(vup, w));
    v = hermes::cuda::cross(w, u);
    lowerLeftCorner = origin - halfWidth * focusDist * u -
                      halfHeight * focusDist * v - focusDist * w;
    horizontal += 2 * halfWidth * focusDist * u;
    vertical += 2 * halfHeight * focusDist * v;
  }
  __host__ __device__ ray getRay(float s, float t) {
    hermes::cuda::vec3 rd = lensRadius * randomInUnitDisk(rng);
    hermes::cuda::vec3 offset = u * rd.x + v * rd.y;
    return ray(origin + offset, lowerLeftCorner + horizontal * s +
                                    vertical * t - origin - offset);
  }
  HS rng;
  hermes::cuda::vec3 u, v, w;
  hermes::cuda::point3 origin, lowerLeftCorner;
  hermes::cuda::vec3 horizontal, vertical;
  float lensRadius;
};

} // namespace heliades

#endif // HELIADES_CORE_CAMERA_H