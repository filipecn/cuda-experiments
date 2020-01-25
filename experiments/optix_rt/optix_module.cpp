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
///\file optix_module.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-22
///
///\brief

#include "optix_module.h"
#include <optix_stubs.h>

namespace optix {

OptixModule Module::handle() const { return module_; }

void Module::createFromPTX(const Context &context, const Pipeline &pipeline,
                           const std::string &PTX, char *log_string,
                           size_t *log_string_size) {
  CHECK_OPTIX(optixModuleCreateFromPTX(
      context.handle(), &compile_options, &pipeline.compile_options,
      PTX.c_str(), PTX.size(), log_string, log_string_size, &module_));
}

} // namespace optix
