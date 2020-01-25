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
///\file optix_module.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-22
///
///\brief

#ifndef OPTIX_MODULE_H
#define OPTIX_MODULE_H

#include "optix_pipeline.h"
#include <string>

namespace optix {

/// Programs are first compiled into modules. One or more modules are then used
/// to create an OptixProgramGroup. Those program groups are then linked into an
/// OptixPipeline to enable them to work together on the GPU.
/// Note: Module lifetimes need to extend to the lifetimes of ProgramGroups that
/// reference them.
class Module {
public:
  OptixModule handle() const;
  /// \param context **[in]** optix context.
  /// \param pipeline **[in]** optix pipeline (only compile options are needed
  /// here).
  /// \param PTX **[in]** programs specified using the Parallel Thread Execution
  /// instruction set (PTX).
  /// \param log_string **[out, default = nullptr]** This string is used to
  /// report information about any compilation that may have occurred, such as
  /// compile errors or verbose information about the compilation result.
  /// \param log_string_size **[out, default = nullptr]** size of the log
  /// message.
  void createFromPTX(const Context &context, const Pipeline &pipeline,
                     const std::string &PTX, char *log_string = nullptr,
                     size_t *log_string_size = nullptr);

  OptixModuleCompileOptions compile_options;

private:
  OptixModule module_;
};

} // namespace optix

#endif // OPTIX_MODULE_H