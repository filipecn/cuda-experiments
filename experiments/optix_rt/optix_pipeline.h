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
///\file optix_pipeline.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-22
///
///\brief

#ifndef OPTIX_PIPELINE_H
#define OPTIX_PIPELINE_H

#include "optix_context.h"
#include "optix_program_group.h"

namespace optix {

/// A pipeline contains all of the programs required for a particular ray
/// tracing launch. An application may use a different pipeline for each
/// launch, or may combine multiple ray-generation programs into a single
/// pipeline.
class Pipeline {
public:
  OptixPipeline handle() const;
  ///\param program_group_set **[in]**
  void add(const ProgramGroupSet &program_group_set);
  ///\brief
  ///
  ///\param context **[in]** optix context
  /// \param log_string **[out, default = nullptr]** This string is used to
  /// report information about any compilation that may have occurred, such as
  /// compile errors or verbose information about the compilation result.
  /// \param log_string_size **[out, default = nullptr]** size of the log
  /// message.
  void create(const Context &context, char *log_string = nullptr,
              size_t *log_string_size = nullptr);
  ///\brief Set the Stack Size object
  ///
  ///\param direct_callable_stack_size_from_traversal **[in]** The direct stack
  /// size requirement for direct callables invoked from IS or AH.
  ///\param direct_callable_stack_size_from_state **[in]** The direct stack size
  /// requirement for direct callables invoked from RG, MS, or CH.
  ///\param continuation_stack_size **[in]** The continuation stack requirement.
  ///\param max_traversable_graph_depth **[in]** The maximum depth of a
  /// traversable graph passed to trace.
  void setStackSize(unsigned int direct_callable_stack_size_from_traversal,
                    unsigned int direct_callable_stack_size_from_state,
                    unsigned int continuation_stack_size,
                    unsigned int max_traversable_graph_depth);

  /// Pipeline compile options must be identical for all modules used to create
  /// program groups linked in a single pipeline.
  OptixPipelineCompileOptions compile_options;
  OptixPipelineLinkOptions link_options;

private:
  std::vector<OptixProgramGroup> program_groups_;
  OptixPipeline pipeline_;
};

} // namespace optix

#endif // OPTIX_PIPELINE_H