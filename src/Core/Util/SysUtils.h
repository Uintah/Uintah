/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* FileUtils.h 
 * 
 * written by 
 *   Allen Sanderson
 *   Nov 2019
 *   University of Utah
 */

#ifndef SYSUTILS_H
#define SYSUTILS_H 1

#include <string>
#include <vector>

namespace Uintah {

  std::string sysPipeCall(const char* cmd);

  unsigned int sysGetNumNUMANodes();
  unsigned int sysGetNumSockets();
  unsigned int sysGetNumCoresPerSockets();
  unsigned int sysGetNumThreadsPerCore();
  std::vector< unsigned int > sysGetNUMANodeCPUs( unsigned int node);
  
} // End namespace Uintah

#endif // SYSUTILS_H

