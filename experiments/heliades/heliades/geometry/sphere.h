#ifndef HELIADES_GEOMETRY_SPHERE_H
#define HELIADES_GEOMETRY_SPHERE_H

#include <heliades/core/hitable.h>
#include <heliades/core/material.h>
#include <hermes/geometry/cuda_point.h>

namespace heliades {

class sphere : public hitable {
public:
  __host__ __device__ sphere() {}
  __host__ __device__ sphere(const hermes::cuda::point3 &center, float radius,
                             material *m)
      : c(center), r(radius), m(m) {}
  __host__ __device__ bool hit(const ray &ra, float tMin, float tMax,
                               hitRecord &rec) const override {
    hermes::cuda::vec3 oc = ra.o - c;
    float A = hermes::cuda::dot(ra.d, ra.d);
    float B = 2.0 * hermes::cuda::dot(oc, ra.d);
    float C = hermes::cuda::dot(oc, oc) - r * r;
    float discriminant = B * B - 4 * A * C;
    if (discriminant > 0) {
      float temp = (-B - sqrtf(discriminant)) / (2 * A);
      if (temp < tMax && temp > tMin) {
        rec.t = temp;
        rec.p = ra(rec.t);
        rec.normal = (rec.p - c) / r;
        rec.materialPtr = m;
        return true;
      }
      temp = (-B + sqrtf(discriminant)) / (2 * A);
      if (temp < tMax && temp > tMin) {
        rec.t = temp;
        rec.p = ra(rec.t);
        rec.normal = (rec.p - c) / r;
        rec.materialPtr = m;
        return true;
      }
    }
    return false;
  }

  hermes::cuda::point3 c; //!< center
  float r;                //!< radius
  material *m;            //!< material
};

} // namespace heliades

#endif // HELIADES_GEOMETRY_SPHERE_H