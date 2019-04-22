#ifndef HELIADES_CORE_MATERIAL_H
#define HELIADES_CORE_MATERIAL_H

#include <heliades/common/random.h>
#include <heliades/core/hitable.h>
#include <heliades/geometry/ray.h>

namespace heliades {

class material {
public:
  /// \param rIn
  /// \param rec
  /// \param attenuation
  /// \param scattered
  /// \return true
  /// \return false
  __host__ __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec,
                                           hermes::cuda::vec3 &attenuation,
                                           ray &scattered, RNG &rng) const = 0;
};

class lambertian : public material {
public:
  __host__ __device__ lambertian(const hermes::cuda::vec3 &a) : albedo(a) {}
  __host__ __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec,
                                           hermes::cuda::vec3 &attenuation,
                                           ray &scattered,
                                           RNG &rng) const override {
    auto target = rec.p + rec.normal + rng.pointInUnitSphere();
    scattered = ray(rec.p, target - rec.p);
    attenuation = albedo;
    return true;
  }
  hermes::cuda::vec3 albedo;
};

class metal : public material {
public:
  __host__ __device__ metal(const hermes::cuda::vec3 &a, float f) : albedo(a) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
  }
  __host__ __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec,
                                           hermes::cuda::vec3 &attenuation,
                                           ray &scattered,
                                           RNG &rng) const override {
    auto reflected =
        hermes::cuda::reflect(hermes::cuda::normalize(rIn.d), rec.normal);
    scattered = ray(rec.p, fuzz * rng.pointInUnitSphere() + reflected);
    attenuation = albedo;
    return hermes::cuda::dot(scattered.d, rec.normal) > 0;
  }
  hermes::cuda::vec3 albedo;
  float fuzz;
};

} // namespace heliades

#endif // HELIADES_CORE_MATERIAL_H