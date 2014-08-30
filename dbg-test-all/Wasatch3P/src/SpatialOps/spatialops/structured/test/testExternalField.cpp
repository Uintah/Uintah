/*
 * Copyright (c) 2014 The University of Utah
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

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "testExternalField.h"

using namespace SpatialOps;
using namespace Point;
using namespace ema::cuda;

int main(int argc, char** argv) {
  unsigned int bytes = 128 * 128 * 128;
  double* T1 = (double*) malloc(sizeof(double) * (bytes));
  double* T2 = (double*) malloc(sizeof(double) * (bytes));
  double* T3 = (double*) malloc(sizeof(double) * (bytes));
  const IntVec npts(128, 128, 128);
  const IntVec badpts(2048, 2048, 2048);

  CUDADeviceInterface& CDI = CUDADeviceInterface::self();

  const MemoryWindow window(npts);

  /** Excercise constructors **/
  memset(T1, 1, bytes * sizeof(double));
  try {
    std::cout << "Checking simple allocation, " << bytes * sizeof(double)
              << " bytes: ";
    PointField p(window, T1, InternalStorage, EXTERNAL_CUDA_GPU, 0);
    std::cout << "PASS\n";
  } catch (...) {
    std::cout << "FAIL\n";
    return -1;
  }

  try {
    std::cout << "Checking proper value initialization: \n";
    PointField p(window, T1, InternalStorage, EXTERNAL_CUDA_GPU, 0);
    CDI.memcpy_from((void*) T2,
                    p.field_values(EXTERNAL_CUDA_GPU, 0),
                    p.allocated_bytes(),
                    p.active_device_index());
    for (unsigned int k = 0; k < bytes; ++k) {
      if (T2[k] != T1[k]) {
        throw;
      }
    }
    std::cout << "PASS\n";
  } catch (...) {
    return -1;
  }

  try {
    std::cout << "Checking consumer field initialization: ";
    PointField p(window, T1, InternalStorage, EXTERNAL_CUDA_GPU, 0);

    p.add_consumer(LOCAL_RAM, 0);
    std::cout << "Finished adding consumer\n";
    memcpy(T2, p.field_values(LOCAL_RAM, 0), p.allocated_bytes());
    CDI.memcpy_from((void*) T3,
                    p.field_values( EXTERNAL_CUDA_GPU, p.active_device_index() ),
                    p.allocated_bytes(),
                    p.active_device_index());
    for (unsigned int k = 0; k < bytes; ++k) {
      if (T2[k] != T1[k] || T3[k] != T1[k]) {
        throw;
      }
    }
    std::cout << "PASS\n";
  } catch (...) {
    return -1;
  }

  try {
    std::cout
        << "Attempting to construct field with invalid memory count ( This shouldn't work ): ";
    // try to allocate more points then we have available memory for.
    CDI.update_memory_statistics();
    CUDAMemStats CMS;
    CDI.get_memory_statistics(CMS, 0);
    const IntVec badpts(1, 1, CMS.f / sizeof(double) + 1);
    const MemoryWindow badwindow(badpts);

    PointField fail(badwindow, NULL, InternalStorage, EXTERNAL_CUDA_GPU, 0);
    std::cout << "FAIL\n";
    return -1;
  } catch (std::runtime_error& e) {
    std::cout << "PASS\n";
  }

  try {
    std::cout
        << "Attempting to construct field with invalid field memory type ( this shouldn't work ): ";
    PointField fail(npts, T1, InternalStorage, DEBUG_TEST_OPT, 0);
    std::cout << "FAIL\n";
    return -1;
  } catch (std::runtime_error& e) {
    std::cout << "PASS\n";
  }

  /** Check assignment and comparison operations **/

  try {
    PointField p(window, T1, InternalStorage, EXTERNAL_CUDA_GPU, 0);
    PointField q(window, T1, InternalStorage, LOCAL_RAM, 0);

    memset(T2, rand(), bytes * sizeof(double));
    PointField r(window, T2, InternalStorage, EXTERNAL_CUDA_GPU, 0);

    std::cout << "P, Q, R assignment and comparison testing: ";
    if (!(p == q)) {
      throw;
    }

    if ((p == q) != !(p != q)) {
      throw;
    }

    if (p == r) {
      throw;
    }

    if (q == r) {
      throw;
    }

    q = r;

    if (q != r) {
      throw;
    }

    if ((p == q) || !(p != q)) {
      throw;
    }

    std::cout << "- Compairison - OK\n";

    std::cout << "PASS\n";
  } catch ( ... ) {
    std::cout << "FAIL\n";
    return -1;
  }

  try {
    std::cout << "Checking SpatialFieldPointer usage: ";

    memset(T1, 1, bytes*sizeof(double));
    PointField q(window, T1, InternalStorage, LOCAL_RAM, 0);

    //Construct using just a window with default values for mtype (LOCAL_RAM) and deviceIndex = 0
    SpatFldPtr<PointField> p = SpatialFieldStore::get<PointField>(window);
    //Construct using full field
    SpatFldPtr<PointField> r = SpatialFieldStore::get<PointField>(q);
    SpatFldPtr<PointField> t = SpatialFieldStore::get<PointField>(window, EXTERNAL_CUDA_GPU, 0);

    p->add_consumer(EXTERNAL_CUDA_GPU, 0);
    t->add_consumer(LOCAL_RAM, 0);

    memcpy(r->field_values(), T1, bytes * sizeof(double));
    r->add_consumer(EXTERNAL_CUDA_GPU, 0);

    memcpy(T2, r->field_values(LOCAL_RAM, 0), r->allocated_bytes());
    CDI.memcpy_from(T3,
                    r->field_values( EXTERNAL_CUDA_GPU, r->active_device_index() ),
                    r->allocated_bytes(),
                    r->active_device_index() );

    for (unsigned int k = 0; k < bytes; ++k) {
      if (T2[k] != T1[k] || T3[k] != T1[k]) {
        std::cout << T2[k] << " : " << T1[k] << std::endl;
        std::cout << T3[k] << " : " << T1[k] << std::endl;

        throw std::runtime_error("Returned values do not match!\n");
      }
    }

    p.detach();
    r.detach();
    t.detach();

    std::cout << "PASS\n";
  } catch ( std::runtime_error& e) {
    std::cout << "FAIL\n";
    std::cout << e.what() << std::endl;
    return -1;
  }

  try {
    std::cout << "Checking proper failure conditions ( bad device type ): ";
    SpatFldPtr<PointField> t = SpatialFieldStore::get<PointField>(window, DEBUG_TEST_OPT, 0);
    std::cout << "FAIL\n";
    return -1;
  } catch ( std::runtime_error& e){
    std::cout << "PASS\n";
  }

  try {
    std::cout << "Checking proper failure conditions ( bad device index ): ";
    SpatFldPtr<PointField> t = SpatialFieldStore::get<PointField>(window, EXTERNAL_CUDA_GPU, 75);
    std::cout << "FAIL\n";
    return -1;
  } catch ( std::runtime_error& e){
    std::cout << "PASS\n";
  }

  std::cout << std::endl << "All tests passed: Success\n";
}
