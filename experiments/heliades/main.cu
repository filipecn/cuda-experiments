#include "io.h"

using namespace heliades;

__global__ void __setupScene(hitable **objects, hitable **world, camera **cam) {
  using namespace hermes::cuda;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float R = cosf(3.14159265358979323846f / 4);
    objects[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f,
                            new lambertian(vec3(0.1, 0.2, 0.5)));
    objects[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.f,
                            new lambertian(vec3(0.8, 0.8, 0.0)));
    objects[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f,
                            new metal(vec3(0.8, 0.6, 0.2)));
    objects[3] =
        new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5));
    objects[4] =
        new sphere(vec3(-1.0f, 0.0f, -1.0f), -0.45f, new dielectric(1.5));
    hermes::cuda::vec3 lookFrom(3, 3, 2);
    hermes::cuda::vec3 lookAt(0, 0, -1);
    float distToFocus = (lookFrom - lookAt).length();
    float aperture = 2.0;
    *cam = new camera(lookFrom, lookAt, vec3(0, 1, 0), 20, 2, aperture,
                      distToFocus);
    *world = new hitableList(objects, 5);
  }
}

__global__ void __setupRandomScene(hitable **objects, hitable **world,
                                   camera **cam) {
  using namespace hermes::cuda;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    HS rng;
    objects[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000,
                            new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    objects[i++] = new sphere(vec3(0, 1, 0), 1, new dielectric(1.5));
    objects[i++] =
        new sphere(vec3(-4, 1, 0), 1, new lambertian(vec3(0.4, 0.2, 0.1)));
    objects[i++] =
        new sphere(vec3(4, 1, 0), 1, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    for (int a = -11; a < 11; a++)
      for (int b = -11; b < 11; b++) {
        float chooseMat = rng.random();
        vec3 center(a + 0.9 * rng.random(), 0.2, b + 0.9 * rng.random());
        if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
          if (chooseMat < 0.8)
            objects[i++] =
                new sphere(center, 0.2,
                           new lambertian(vec3(rng.random() * rng.random(),
                                               rng.random() * rng.random(),
                                               rng.random() * rng.random())));
          else if (chooseMat < 0.95)
            objects[i++] =
                new sphere(center, 0.2,
                           new metal(vec3(0.5 * (1 + rng.random()),
                                          0.5 * (1 + rng.random()),
                                          0.5 * (1 + rng.random()))));
          else
            objects[i++] = new sphere(center, 0.2, new dielectric(1.5));
        }
      }
    hermes::cuda::vec3 lookFrom(10, 2, 3);
    hermes::cuda::vec3 lookAt(4, 1, 1);
    float distToFocus = (lookFrom - lookAt).length();
    float aperture = 0.001;
    *cam = new camera(lookFrom, lookAt, vec3(0, 1, 0), 20, 2, aperture,
                      distToFocus);
    *world = new hitableList(objects, 40);
  }
}

__device__ hermes::cuda::vec3 color(const heliades::ray &r, hitable **world,
                                    HS &rng, int level = 0) {
  using namespace hermes::cuda;
  hitRecord rec;
  if (level < 5 &&
      (*world)->hit(r, 0.00001, hermes::cuda::Constants::greatest<float>(),
                    rec)) {
    ray scattered;
    vec3 attenuation;
    if (level < 5 &&
        rec.materialPtr->scatter(r, rec, attenuation, scattered, rng))
      return attenuation * color(scattered, world, rng, level + 1);
    return vec3(0, 0, 0);
  }
  vec3 d = hermes::cuda::normalize(r.d);
  float t = 0.5 * (d.y + 1.0);
  return vec3(1.0f - t) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void __render(heliades::Film::Pixel *out, int w, int h,
                         hitable **world, camera **cam, int ns) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * w + x;
  // printf("%d\n", index);
  if (x < w && y < h) {
    HS rng, sng;
    rng.setIndex(index + 1);
    sng.setIndex(index + 1);
    hermes::cuda::vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
      auto sd = sng.random2();
      float u = float(x + sd.x) / float(w);
      float v = float(y + sd.y) / float(h);
      heliades::ray r = (*cam)->getRay(u, v);
      col += color(r, world, rng);
    }
    col /= float(ns);
    col = hermes::cuda::vec3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
    out[index].xyz[0] = 255.99f * col.r();
    out[index].xyz[1] = 255.99f * col.g();
    out[index].xyz[2] = 255.99f * col.b();
  }
}

int main(int argc, char **argv) {
  // scene
  hitable **objects = nullptr;
  hitable **world = nullptr;
  camera **cam = nullptr;
  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaMalloc(&objects, 501 * sizeof(hitable *)));
    CUDA_CHECK(cudaMalloc(&world, sizeof(hitable *)));
    CUDA_CHECK(cudaMalloc(&cam, sizeof(camera *)));
  }
  __setupRandomScene<<<1, 1>>>(objects, world, cam);
  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // rendering
  hermes::cuda::vec2u imageSize(800, 400);
  heliades::Film film(imageSize);
  hermes::ThreadArrayDistributionInfo td(imageSize.x, imageSize.y);
  __render<<<td.gridSize, td.blockSize>>>(film.pixelsDeviceData(), imageSize.x,
                                          imageSize.y, world, cam, 50);

  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  std::cerr << "render complete\n";
  // cudaFree(scene.list[0]);
  // cudaFree(scene.list[1]);
  // VIS
  circe::SceneApp<> app(imageSize.x, imageSize.y, "", false);
  app.addViewport2D(0, 0, imageSize.x, imageSize.y);
  CudaOpenGLInterop cgl(imageSize.x, imageSize.y);
  pixelsToTexture(imageSize.x, imageSize.y, film.pixelsDeviceData(),
                  cgl.bufferPointer());
  cgl.sendToTexture();
  circe::ScreenQuad screen;
  app.renderCallback = [&]() {
    screen.shader->begin();
    screen.shader->setUniform("tex", 0);
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
  };
  app.run();
  return 0;
}