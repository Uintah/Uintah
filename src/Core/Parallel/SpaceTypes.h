/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

// The purpose of this file is to provide an enum that can at runtime be used to 
// determine which execution space and memory space a task is assigned.  

#ifndef UINTAH_CORE_PARALLEL_SPACE_TYPES_H
#define UINTAH_CORE_PARALLEL_SPACE_TYPES_H

enum TaskAssignedExecutionSpace {
  NONE_EXECUTION_SPACE = 0,
  UINTAH_CPU = 1,                          //binary 001
  KOKKOS_OPENMP = 2,                       //binary 010
  KOKKOS_CUDA = 4,                         //binary 100
};

enum TaskAssignedMemorySpace {
  NONE_MEMORY_SPACE = 0,
  UINTAH_HOSTSPACE = 1,                          //binary 001
  KOKKOS_HOSTSPACE = 2,                       //binary 010
  KOKKOS_CUDASPACE = 4,                         //binary 100
};

#endif // UINTAH_CORE_PARALLEL_SPACE_TYPES_H
