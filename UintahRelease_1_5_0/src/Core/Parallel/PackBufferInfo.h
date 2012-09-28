/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef UINTAH_HOMEBREW_PackBufferInfo_H
#define UINTAH_HOMEBREW_PackBufferInfo_H

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <Core/Parallel/BufferInfo.h>
#include <Core/Util/RefCounted.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
  
  class PackedBuffer : public RefCounted
  {
  public:
    PackedBuffer(int bytes)
      : buf((void*)(scinew char[bytes])), bufsize(bytes) {}
    ~PackedBuffer()
    { delete[] (char*)buf; buf=0; }
    void* getBuffer() { return buf; }
    int getBufSize() { return bufsize; }
  private:
    void* buf;
    int bufsize;
  };

  class PackBufferInfo : public BufferInfo {
  public:
    PackBufferInfo();
    ~PackBufferInfo();

    void get_type(void*&, int&, MPI_Datatype&, MPI_Comm comm);
    void get_type(void*&, int&, MPI_Datatype&);
    void pack(MPI_Comm comm, int& out_count);
    void unpack(MPI_Comm comm, MPI_Status &status); 
    // PackBufferInfo is to be an AfterCommuncationHandler object for the
    // MPI_CommunicationRecord template in MPIScheduler.cc.  After receive
    // requests have finished, then it needs to unpack what got received.
   void finishedCommunication(const ProcessorGroup * pg,MPI_Status &status)
    { unpack(pg->getComm(),status); }
    
  private:
    PackBufferInfo(const PackBufferInfo&);
    PackBufferInfo& operator=(const PackBufferInfo&);

    PackedBuffer* packedBuffer;
  };
}

#endif
