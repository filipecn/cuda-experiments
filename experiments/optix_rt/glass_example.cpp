#include "renderer.h"

int main(int argc, char **argv) {
  Renderer renderer;
  renderer.resize(1024, 1024);
  renderer.render();
  return 0;
}