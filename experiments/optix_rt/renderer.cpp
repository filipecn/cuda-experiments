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
///\file renderer.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-11
///
///\brief

#include "renderer.h"
#include <iostream>
#include <optix_stubs.h>

extern "C" char embedded_ptx_code[];

Renderer::Renderer() {
  createModule();
  createPrograms();
  createPipeline();
  buildSBT();
}

void Renderer::resize(uint32_t width, uint32_t height) {
  framebuffer_.resize(width * height);
  framebuffer_.allocate();
  launch_params_.fbSize = hermes::cuda::vec2u(width, height);
  launch_params_.colorBuffer = framebuffer_.ptr();
}

void Renderer::downloadPixels(uint32_t pixels[]) const {}

void Renderer::createModule() {
  optix_module_.compile_options.maxRegisterCount = 50;
  optix_module_.compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  optix_module_.compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  optix_pipeline_.compile_options = {};
  optix_pipeline_.compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  optix_pipeline_.compile_options.usesMotionBlur = false;
  optix_pipeline_.compile_options.numPayloadValues = 2;
  optix_pipeline_.compile_options.numAttributeValues = 2;
  optix_pipeline_.compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  optix_pipeline_.compile_options.pipelineLaunchParamsVariableName =
      "optixLaunchParams";

  optix_pipeline_.link_options.overrideUsesMotionBlur = false;
  optix_pipeline_.link_options.maxTraceDepth = 2;

  const std::string ptxCode = embedded_ptx_code;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  optix_module_.createFromPTX(optix_context_, optix_pipeline_, ptxCode, log,
                              &sizeof_log);
  if (sizeof_log > 1)
    std::cerr << log << std::endl;
}

void Renderer::createPrograms() {
  char log[2048];
  size_t sizeof_log = sizeof(log);
  { // ray gen programs
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module = optix_module_.handle();
    pg_desc.raygen.entryFunctionName = "__raygen__renderFrame";
    raygen_.add(pg_options, pg_desc);
    raygen_.create(optix_context_, log, &sizeof_log);
    if (sizeof_log > 1)
      std::cerr << log << std::endl;
  }
  { // miss programs
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module = optix_module_.handle();
    pg_desc.miss.entryFunctionName = "__miss__radiance";
    miss_.add(pg_options, pg_desc);
    miss_.create(optix_context_, log, &sizeof_log);
    if (sizeof_log > 1)
      std::cerr << log << std::endl;
  }
  { // hitgroup programs
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = optix_module_.handle();
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pg_desc.hitgroup.moduleAH = optix_module_.handle();
    pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    hitgroup_.add(pg_options, pg_desc);
    hitgroup_.create(optix_context_, log, &sizeof_log);
    if (sizeof_log > 1)
      std::cerr << log << std::endl;
  }
}

void Renderer::createPipeline() {
  optix_pipeline_.add(raygen_);
  optix_pipeline_.add(miss_);
  optix_pipeline_.add(hitgroup_);
  char log[2048];
  size_t sizeof_log = sizeof(log);
  optix_pipeline_.create(optix_context_, log, &sizeof_log);
  if (sizeof_log > 1)
    std::cerr << log << std::endl;
  optix_pipeline_.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, 1);
}

void Renderer::buildSBT() {
  using namespace hermes::cuda;
  // raygen records
  MemoryBlock1<MemoryLocation::HOST, RaygenRecord> h_raygen_records(
      raygen_.size(), {});
  for (size_t i = 0; i < raygen_.size(); ++i) {
    RaygenRecord rec;
    CHECK_OPTIX(optixSbtRecordPackHeader(raygen_.programGroups()[i], &rec));
    rec.data = nullptr; /* for now ... */
    h_raygen_records.accessor()[i] = rec;
  }
  raygen_records_.resize(raygen_.size());
  raygen_records_.allocate();
  memcpy(raygen_records_, h_raygen_records);
  // miss records
  MemoryBlock1<MemoryLocation::HOST, MissRecord> h_miss_records(miss_.size(),
                                                                {});
  for (size_t i = 0; i < miss_.size(); ++i) {
    MissRecord rec;
    CHECK_OPTIX(optixSbtRecordPackHeader(miss_.programGroups()[i], &rec));
    rec.data = nullptr; /* for now ... */
    h_miss_records.accessor()[i] = rec;
  }
  miss_records_.resize(miss_.size());
  miss_records_.allocate();
  memcpy(miss_records_, h_miss_records);
  // hitgroup records
  MemoryBlock1<MemoryLocation::HOST, HitgroupRecord> h_hitgroup_records(
      hitgroup_.size(), {});
  for (size_t i = 0; i < 1 /*hitgroup_.size()*/; ++i) {
    HitgroupRecord rec;
    CHECK_OPTIX(optixSbtRecordPackHeader(hitgroup_.programGroups()[i], &rec));
    rec.object_id = i;
    h_hitgroup_records.accessor()[i] = rec;
  }
  hitgroup_records_.resize(miss_.size());
  hitgroup_records_.allocate();
  memcpy(hitgroup_records_, h_hitgroup_records);

  sbt.raygenRecord = (CUdeviceptr)raygen_records_.ptr();
  sbt.missRecordBase = (CUdeviceptr)miss_records_.ptr();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)miss_records_.size();
  sbt.hitgroupRecordBase = (CUdeviceptr)hitgroup_records_.ptr();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroup_records_.size();
}

void Renderer::render() {
  using namespace hermes::cuda;
  MemoryBlock1<MemoryLocation::HOST, LaunchParams> h_launch_params(
      1, launch_params_);
  d_launch_params_.resize(1);
  d_launch_params_.allocate();
  memcpy(d_launch_params_, h_launch_params);
  launch_params_.frameID++;
  CHECK_OPTIX(optixLaunch(optix_pipeline_.handle(), optix_context_.stream(),
                          (CUdeviceptr)d_launch_params_.ptr(),
                          d_launch_params_.memorySize(), &sbt,
                          launch_params_.fbSize.x, launch_params_.fbSize.y, 1));
  cudaDeviceSynchronize();
}
