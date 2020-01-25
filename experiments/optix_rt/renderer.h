/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file Renderer.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-11
///
///\brief

#ifndef OPTIX_RENDERER_H
#define OPTIX_RENDERER_H

#include "launch_params.h"
#include "optix_module.h"
#include "optix_program_group.h"
#include <hermes/hermes.h>

class Renderer {
public:
  Renderer();
  void resize(uint32_t width, uint32_t height);
  void downloadPixels(uint32_t pixels[]) const;
  void render();

private:
  void createModule();
  void createPrograms();
  void createPipeline();
  void buildSBT();

  optix::Context optix_context_;
  optix::Pipeline optix_pipeline_;
  optix::Module optix_module_;

  optix::ProgramGroupSet raygen_;
  optix::ProgramGroupSet miss_;
  optix::ProgramGroupSet hitgroup_;

  /*! SBT record for a raygen program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };
  /*! SBT record for a miss program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };
  /*! SBT record for a hitgroup program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int object_id;
  };
  hermes::cuda::MemoryBlock1<hermes::cuda::MemoryLocation::DEVICE, RaygenRecord>
      raygen_records_;
  hermes::cuda::MemoryBlock1<hermes::cuda::MemoryLocation::DEVICE, MissRecord>
      miss_records_;
  hermes::cuda::MemoryBlock1<hermes::cuda::MemoryLocation::DEVICE,
                             HitgroupRecord>
      hitgroup_records_;
  OptixShaderBindingTable sbt = {};

  LaunchParams launch_params_;
  hermes::cuda::MemoryBlock1<hermes::cuda::MemoryLocation::DEVICE, LaunchParams>
      d_launch_params_;
  hermes::cuda::MemoryBlock1<hermes::cuda::MemoryLocation::DEVICE, uint32_t>
      framebuffer_;
};

#endif