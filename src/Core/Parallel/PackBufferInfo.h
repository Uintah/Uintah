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

#ifndef CORE_PARALLEL_PACKBUFFERINFO_H
#define CORE_PARALLEL_PACKBUFFERINFO_H

#include <Core/Parallel/BufferInfo.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahMPI.h>
#include <Core/Util/RefCounted.h>

namespace Uintah {


class PackedBuffer : public RefCounted {

public:
  
  PackedBuffer(int bytes) : m_buffer((void*)(scinew char[bytes])), m_buffer_size(bytes) {}

  ~PackedBuffer() {
    delete[] (char*)m_buffer;
    m_buffer = nullptr;
  }

  void * getBuffer() { return m_buffer; }

  int getBufSize()  { return m_buffer_size; }

private:
  void * m_buffer{nullptr};
  int    m_buffer_size{0};
};


class PackBufferInfo : public BufferInfo {

  public:

    PackBufferInfo();

    ~PackBufferInfo();

    void get_type( void         *& out_buf
                 , int&            out_count
                 , MPI_Datatype  & out_datatype
                 , MPI_Comm        comm
                 );

    void get_type( void         *&
                 , int           &
                 , MPI_Datatype  &
                 );

    void pack( MPI_Comm comm, int & out_count );

    void unpack( MPI_Comm comm, MPI_Status & status );

    // PackBufferInfo is to be an AfterCommuncationHandler object for the
    // MPI_CommunicationRecord template in MPIScheduler.cc.  After receive
    // requests have finished, then it needs to unpack what got received.
    void finishedCommunication( const ProcessorGroup* pg, MPI_Status& status )
    {
      unpack( pg->getComm(), status );
    }


  private:

    // disable copy and assignment
    PackedBuffer * m_packed_buffer{nullptr};

    // eliminate copy, assignment and move
    PackBufferInfo( const PackBufferInfo & )            = delete;
    PackBufferInfo& operator=( const PackBufferInfo & ) = delete;
    PackBufferInfo( PackBufferInfo && )                 = delete;
    PackBufferInfo& operator=( PackBufferInfo && )      = delete;

}; // PackBufferInfo

} // end namespace Uintah

#endif // CORE_PARALLEL_PACKBUFFERINFO_H
