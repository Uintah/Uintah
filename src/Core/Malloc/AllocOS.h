/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


/*
 *  AllocOS.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#ifndef CORE_MALLOC_ALLOCOS_H
#define CORE_MALLOC_ALLOCOS_H

#include <cstdlib>

namespace Uintah {

struct Allocator;

struct OSHunk {

  static OSHunk* alloc( size_t size, bool returnable, Allocator* allocator );

  static void free( OSHunk* );

  void* data;
  OSHunk* next;

  int      ninuse;
  size_t   spaceleft;
  void   * curr;
  size_t   len;
  size_t   alloc_len;
  bool     returnable;
};

} // End namespace Uintah

#endif // CORE_MALLOC_ALLOCOS_H
