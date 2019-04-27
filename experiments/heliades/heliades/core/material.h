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
  __host__ __device__ metal(const hermes::cuda::vec3 &a, float f = 0)
      : albedo(a) {
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

inline __host__ __device__ bool refract(const hermes::cuda::vec3 &v,
                                        const hermes::cuda::vec3 &n,
                                        float ni_nt,
                                        hermes::cuda::vec3 &refracted) {
  hermes::cuda::vec3 uv = hermes::cuda::normalize(v);
  float dt = hermes::cuda::dot(uv, n);
  float discriminant = 1. - ni_nt * ni_nt * (1. - dt * dt);
  if (discriminant > 0) {
    refracted = ni_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }
  return false;
}

inline __host__ __device__ float schlick(float cosine, float refIdx) {
  float r0 = (1 - refIdx) / (1 + refIdx);
  r0 *= r0;
  return r0 + (1 - r0) * powf(1 - cosine, 5);
}

class dielectric : public material {
public:
  __host__ __device__ dielectric(float ri) : refIdx(ri) {}
  __host__ __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec,
                                           hermes::cuda::vec3 &attenuation,
                                           ray &scattered,
                                           RNG &rng) const override {
    using namespace hermes::cuda;
    vec3 outwardNormal;
    float ni_nt;
    attenuation = vec3(1, 1, 1);
    vec3 refracted;
    float reflectProb;
    float cosine;
    if (dot(rIn.d, rec.normal) > 0) {
      outwardNormal = -rec.normal;
      ni_nt = refIdx;
      cosine = refIdx * dot(rIn.d, rec.normal) / rIn.d.length();
    } else {
      outwardNormal = rec.normal;
      ni_nt = 1 / refIdx;
      cosine = -dot(rIn.d, rec.normal) / rIn.d.length();
    }
    if (refract(rIn.d, outwardNormal, ni_nt, refracted))
      reflectProb = schlick(cosine, refIdx);
    else
      reflectProb = 1;
    if (rng.random() < reflectProb) {
      vec3 reflected = hermes::cuda::reflect(rIn.d, rec.normal);
      scattered = ray(rec.p, reflected);
    } else
      scattered = ray(rec.p, refracted);
    return true;
  }

  float refIdx;
};

} // namespace heliades

#endif // HELIADES_CORE_MATERIAL_H