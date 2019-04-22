#ifndef HELIADES_COMMON_RANDOM_H
#define HELIADES_COMMON_RANDOM_H

#include <hermes/common/cuda_random.h>
#include <hermes/geometry/cuda_point.h>

namespace heliades {

class RNG {
public:
  virtual __host__ __device__ void setIndex(size_t i) {}
  virtual __host__ __device__ float random() = 0;
  virtual __host__ __device__ hermes::cuda::vec2 random2() = 0;
  virtual __host__ __device__ hermes::cuda::vec3 random3() = 0;
  virtual __host__ __device__ hermes::cuda::point3 pointInUnitSphere() {
    hermes::cuda::vec3 p;
    do {
      p = 2.f * random3() - hermes::cuda::vec3(1, 1, 1);
    } while (p.length2() >= 1.0);
    return hermes::cuda::point3(p.x, p.y, p.z);
  }
};

class HS : public RNG {
public:
  __host__ __device__ HS(uint a = 3, uint b = 5, uint c = 7) {
    hs[0].setBase(a);
    hs[1].setBase(b);
    hs[2].setBase(c);
  }
  __host__ __device__ void setIndex(size_t i) {
    hs[0].setIndex(i);
    hs[1].setIndex(i);
    hs[2].setIndex(i);
  }
  __host__ __device__ float random() override { return hs[0].randomFloat(); }
  __host__ __device__ hermes::cuda::vec2 random2() override {
    return hermes::cuda::vec2(hs[0].randomFloat(), hs[1].randomFloat());
  }
  __host__ __device__ hermes::cuda::vec3 random3() override {
    return hermes::cuda::vec3(hs[0].randomFloat(), hs[1].randomFloat(),
                              hs[2].randomFloat());
  }

private:
  hermes::cuda::HaltonSequence hs[3];
};

} // namespace heliades

#endif