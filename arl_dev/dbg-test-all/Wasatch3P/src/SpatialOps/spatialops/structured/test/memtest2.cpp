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
#include <string>
#include <stdexcept>
#include <stdio.h>

#define DEBUG_CUDA_VERBOSE
#include <spatialops/structured/ExternalAllocators.h>

using namespace ema::cuda;

int main(int argc, char** argv) {
  CUDADeviceInterface& CDI = CUDADeviceInterface::self();
  CUDASharedPointer p;
  int temp;
  bool should_fail = false;

  try {
  //going to CDI.get_device_count() causes problems on machines where not all devices are free for use
  for (int device = 0; device < 1; ++device) {
      CDI.update_memory_statistics();
      CUDAMemStats CMS;
      CDI.get_memory_statistics(CMS, device);

      std::cout << "Testing pointer creation, allocating " << CMS.f
          << " bytes, on device " << device << "...";
      p = CDI.get_shared_pointer((CMS.f / 2), device);

      if (p.get_refcount() != 1) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found invalid reference count on pointer P for device: "
            << device;
        throw(std::runtime_error(msg.str()));
      }

      if (p.get_deviceID() != device) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found incorrect deviceID on pointer P for device: " << device;
        throw(std::runtime_error(msg.str()));
      }
      std::cout << " PASS\n";

      std::cout << "Testing pointer assignment operations...";
      CUDASharedPointer q;
      q = p;

      if (q.get_refcount() != 2) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found invalid reference count, " << q.get_refcount()
                << ", on pointer Q for device: " << device;
        throw(std::runtime_error(msg.str()));
      }

      if (q.get_deviceID() != device) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found incorrect deviceID on pointer Q for device: " << device;
        throw(std::runtime_error(msg.str()));
      }
      std::cout << " PASS\n";

      std::cout << "Testing pointer detach and reassignment...";
      q.detach();

      if (p.get_refcount() != 1) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found invalid reference count, " << p.get_refcount()
                << ", on pointer P for device: " << device;
        throw(std::runtime_error(msg.str()));
      }

      if (p.get_deviceID() != device) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found incorrect deviceID on pointer P for device: " << device;
        throw(std::runtime_error(msg.str()));
      }

      for (int i = 0; i < 10; i++) {
        CUDASharedPointer x;
        x = p;
        CUDASharedPointer y = x;
        CUDASharedPointer z(y);
      }

      if (p.get_refcount() != 1) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found invalid reference count," << p.get_refcount()
                << ", on pointer P for device: " << device;
        throw(std::runtime_error(msg.str()));
      }

      if (p.get_deviceID() != device) {
        std::cout << " FAIL\n";
        std::ostringstream msg;
        msg << "Found incorrect deviceID on pointer P for device: " << device;
        throw(std::runtime_error(msg.str()));
      }
      std::cout << " PASS\n";
    }
  }
  catch( std::runtime_error& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  printf("Success\n");

  return 0;
}
