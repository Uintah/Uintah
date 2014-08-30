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
#include <string>
#include <stdexcept>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define DEBUG_CUDA_VERBOSE
#include <spatialops/structured/ExternalAllocators.h>

using namespace ema::cuda;

int main(int argc, char** argv) {
  CUDADeviceInterface& CDI = CUDADeviceInterface::self();
  CUDASharedPointer p;
  int temp;

  //going to CDI.get_device_count() causes problems on machines where not all devices are free for use
  for (int device = 0; device < 1; ++device) {
    try {
      for (int i = 1;; i *= 2) {
        CDI.update_memory_statistics();
        CUDAMemStats CMS;
        CDI.get_memory_statistics(CMS, device);

        std::cout << "Attempting allocation of size " << i << " bytes, on device " << device;
        std::cout << "\n\t Free memory: " << CMS.f << " / " << CMS.t << std::endl;

        p = CDI.get_shared_pointer(i, 0);
        char* bytesin = (char*)malloc(sizeof(char)*i);
        char* bytesout = (char*)malloc(sizeof(char)*i);

        bzero(bytesin, i);
        bzero(bytesout, i);
        std::cout << "Checking zeros read/write... ";
        CDI.memcpy_to(p, (void*)bytesin, i);
        CDI.memcpy_from((void*)bytesout, p, i);

        if( memcmp(bytesin, bytesout, i) ){
          std::cout << "failed -> Zero byte pattern does do not match\n";
          exit(1);
        }

        std::cout << "OK\n";

        memset(bytesin, 1, i);
        bzero(bytesout, i);
        std::cout << "Checking ones read/write... ";
        CDI.memcpy_to(p, (void*)bytesin, i);
        CDI.memcpy_from((void*)bytesout, p, i);

        if( memcmp(bytesin, bytesout, i) ){
          std::cout << "failed -> Ones byte pattern does not match\n";
          exit(1);
        }

        std::cout << "OK\n";

        srand(0);
        for(int k = 0; i < i; ++k){
          bytesin[k] = rand();
        }
        bzero(bytesout, i);
        std::cout << "Checking random read/write... ";
        CDI.memcpy_to(p, (void*)bytesin, i);
        CDI.memcpy_from((void*)bytesout, p, i);

        if( memcmp(bytesin, bytesout, i) ){
          std::cout << "failed -> Random byte pattern does not match\n";
          exit(1);
        }

        std::cout << "OK\n";
      }
    }
    catch ( std::runtime_error& e ) {
      //Note: malloc will fail at some point, this is expected
      std::cout << e.what() << std::endl;
    }
  }

  printf("Success\n");

  return 0;
}
