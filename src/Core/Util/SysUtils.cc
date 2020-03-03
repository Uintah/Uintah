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

/* SysUtils.cc
 * 
 * written by 
 *   Allen Sanderson
 *   Nov 2019
 *   University of Utah
 */

#include <Core/Util/SysUtils.h>

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace Uintah {

  std::string sysPipeCall(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    
    if (!pipe)
      throw std::runtime_error("popen() failed!");
    
    while (!feof(pipe.get())) {
      if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
        result += buffer.data();
    }
    return result;
  }

#ifdef __linux
  unsigned int sysGetNumNUMANodes() {
    return std::stoi( sysPipeCall("lscpu | grep 'NUMA node(s):' | sed -n '${s/.* //; p}'") );
  }
  
  unsigned int sysGetNumSockets() {
    return std::stoi( sysPipeCall("lscpu | grep 'Socket(s):' | sed -n '${s/.* //; p}'") );
  }  

  unsigned int sysGetNumCoresPerSockets() {
    return std::stoi( sysPipeCall("lscpu | grep 'Core(s) per socket:' | sed -n '${s/.* //; p}'") );
  }  

  unsigned int sysGetNumThreadsPerCore() {
    return std::stoi( sysPipeCall("lscpu | grep 'Thread(s) per core:' | sed -n '${s/.* //; p}'") );
  }
  
  std::vector< unsigned int > sysGetNUMANodeCPUs( unsigned int node) {
    std::string cpusStr = sysPipeCall( std::string( "lscpu | grep 'NUMA node" +
                                                    std::to_string( node ) +
                                                    " CPU(s):' | sed -n '${s/.* //; p}'" ).c_str());

    std::vector< unsigned int > cpus;
    
    while( cpusStr.size() ) {
      size_t found = cpusStr.find( "," );

      cpus.push_back( std::stoi( cpusStr.substr(0, found) ) );
      cpusStr = cpusStr.substr(found+1);
    }

    return cpus;
  }

#elif defined __APPLE__

  unsigned int sysGetNumNUMANodes() {
    // OS X does not support NUMA.
    return 1;
  }
  
  unsigned int sysGetNumSockets() {
    // Currently no multi-socket OS X machines.
    return 1;
  }  

  unsigned int sysGetNumCoresPerSockets() {
    return std::stoi( sysPipeCall("sysctl -a | grep machdep.cpu.core_count | sed -n 's/.* //; p'") );
  }  

  unsigned int sysGetNumThreadsPerCore() {
    return std::stoi( sysPipeCall("sysctl -a | grep machdep.cpu.thread_count | sed -n 's/.* //; p'") ) / sysGetNumCoresPerSockets();
  }
  
  std::vector< unsigned int > sysGetNUMANodeCPUs( unsigned int node) {

    // OS X does not support NUMA so just enter the CPU ids based on
    // the thread count.
    std::vector< unsigned int > cpus;
                          
    unsigned int nThreads =
      std::stoi( sysPipeCall("sysctl -a | grep machdep.cpu.thread_count") );

    for( unsigned int i=0; i<nThreads; ++i )
         cpus.push_back( i );
    return cpus;
  }
#endif
  
} // End namespace Uintah
