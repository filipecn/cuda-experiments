#ifndef HELIADES_CORE_HITABLE_H
#define HELIADES_CORE_HITABLE_H

#include <heliades/geometry/ray.h>
#include <hermes/geometry/cuda_point.h>

namespace heliades {

class material;
/// Ray intersection point info
struct hitRecord {
  float t;                   ///!<  ray's parametric coordinate
  hermes::cuda::point3 p;    ///!< intersection point
  hermes::cuda::vec3 normal; ///!< surface's normal at p
  material *materialPtr;
};

/// Interface for light ray interacting objects.
class hitable {
public:
  /// \param r light ray
  /// \param tMin
  /// \param tMax
  /// \param rec
  /// \return true
  __host__ __device__ virtual bool hit(const ray &r, float tMin, float tMax,
                                       hitRecord &rec) const = 0;
};

class hitableList : public hitable {
public:
  __host__ __device__ hitableList() {}
  __host__ __device__ hitableList(hitable **l, int n) : list(l), size(n) {}
  __host__ __device__ virtual bool hit(const ray &r, float tMin, float tMax,
                                       hitRecord &rec) const {
    hitRecord tmpRec;
    bool hitAnything = false;
    double closestSoFar = tMax;
    for (int i = 0; i < size; ++i) {
      if (list[i]->hit(r, tMin, closestSoFar, tmpRec)) {
        hitAnything = true;
        closestSoFar = tmpRec.t;
        rec = tmpRec;
      }
    }
    return hitAnything;
  }
  hitable **list = nullptr;
  int size = 0;
};

} // namespace heliades

#endif