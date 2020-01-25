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
///\file optix_pipeline.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-22
///
///\brief

#include "optix_pipeline.h"
#include <optix_stubs.h>

namespace optix {

OptixPipeline Pipeline::handle() const { return pipeline_; }

void Pipeline::add(const ProgramGroupSet &program_group_set) {
  for (auto pg : program_group_set.programGroups())
    program_groups_.push_back(pg);
}

void Pipeline::create(const Context &context, char *log_string,
                      size_t *log_string_size) {
  CHECK_OPTIX(optixPipelineCreate(
      context.handle(), &compile_options, &link_options, &program_groups_[0],
      program_groups_.size(), log_string, log_string_size, &pipeline_));
}

void Pipeline::setStackSize(
    unsigned int direct_callable_stack_size_from_traversal,
    unsigned int direct_callable_stack_size_from_state,
    unsigned int continuation_stack_size,
    unsigned int max_traversable_graph_depth) {
  CHECK_OPTIX(optixPipelineSetStackSize(
      pipeline_, direct_callable_stack_size_from_traversal,
      direct_callable_stack_size_from_state, continuation_stack_size,
      max_traversable_graph_depth));
}

} // namespace optix