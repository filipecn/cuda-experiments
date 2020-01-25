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
///\file optix_program_group.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-24
///
///\brief

#ifndef OPTIX_PROGRAM_GROUP_H
#define OPTIX_PROGRAM_GROUP_H

#include "optix_context.h"
#include <vector>

namespace optix {

/// OptixProgramGroup objects are created from one to three OptixModule objects
/// and are used to fill the header of the SBT records. Program groups can also
/// be used across pipelines as long as the compilation options match.
class ProgramGroupSet {
public:
  ProgramGroupSet();
  size_t size() const;
  const std::vector<OptixProgramGroup> &programGroups() const;
  ///\param pg_options **[in]**
  ///\param pg_desc **[in]**
  ///\return size_t
  size_t add(OptixProgramGroupOptions pg_options,
             OptixProgramGroupDesc pg_desc);
  ///\param program_group_id **[in]**
  ///\return OptixStackSizes
  OptixStackSizes stackSizes(size_t program_group_id) const;
  ///\param context **[in]** optix context
  /// \param log_string **[out, default = nullptr]** This string is used to
  /// report information about any compilation that may have occurred, such as
  /// compile errors or verbose information about the compilation result.
  /// \param log_string_size **[out, default = nullptr]** size of the log
  /// message.
  void create(const Context &context, char *log_string = nullptr,
              size_t *log_string_size = nullptr);

private:
  std::vector<OptixProgramGroup> program_groups_;
  std::vector<OptixProgramGroupOptions> pg_options_;
  std::vector<OptixProgramGroupDesc> pg_descs_;
};

} // namespace optix

#endif