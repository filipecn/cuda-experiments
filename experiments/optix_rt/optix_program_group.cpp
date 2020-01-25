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
///\file optix_program_group.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-24
///
///\brief

#include "optix_program_group.h"
#include <optix_stubs.h>

namespace optix {

ProgramGroupSet::ProgramGroupSet() = default;

size_t ProgramGroupSet::size() const { return program_groups_.size(); }

const std::vector<OptixProgramGroup> &ProgramGroupSet::programGroups() const {
  return program_groups_;
}

OptixStackSizes ProgramGroupSet::stackSizes(size_t program_group_id) const {
  OptixStackSizes stack_sizes{};
  CHECK_OPTIX(optixProgramGroupGetStackSize(program_groups_[program_group_id],
                                            &stack_sizes));
  return stack_sizes;
}

size_t ProgramGroupSet::add(OptixProgramGroupOptions pg_options,
                            OptixProgramGroupDesc pg_desc) {
  pg_options_.push_back(pg_options);
  pg_descs_.push_back(pg_desc);
  return pg_options_.size() - 1;
}

void ProgramGroupSet::create(const Context &context, char *log_string,
                             size_t *log_string_size) {
  program_groups_.resize(pg_options_.size());
  CHECK_OPTIX(optixProgramGroupCreate(
      context.handle(), &pg_descs_[0], pg_descs_.size(), &pg_options_[0],
      log_string, log_string_size, &program_groups_[0]));
}

} // namespace optix