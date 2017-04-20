/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/Assert.h>
#include <Core/Util/RefCounted.h>

using namespace Uintah;

#include <iostream>
#include <string.h>


//_____________________________________________________________________________
//
PackBufferInfo::PackBufferInfo() : BufferInfo() {}

//_____________________________________________________________________________
//
PackBufferInfo::~PackBufferInfo()
{
  if (m_packed_buffer && m_packed_buffer->removeReference()) {
    delete m_packed_buffer;
    m_packed_buffer = nullptr;
  }
}

//_____________________________________________________________________________
//
void
PackBufferInfo::get_type( void         *& out_buf
                        , int&            out_count
                        , MPI_Datatype  & out_datatype
                        , MPI_Comm        comm
                        )
{
  ASSERT(count() > 0);
  if (!m_have_datatype) {
    int packed_size;
    int total_packed_size = 0;
    for (unsigned int i = 0; i < m_start_bufs.size(); i++) {
      if (m_counts[i] > 0) {
        Uintah::MPI::Pack_size(m_counts[i], m_datatypes[i], comm, &packed_size);
        total_packed_size += packed_size;
      }
    }

    m_packed_buffer = scinew PackedBuffer(total_packed_size);
    m_packed_buffer->addReference();

    m_datatype = MPI_PACKED;
    m_count = total_packed_size;
    m_buffer = m_packed_buffer->getBuffer();
    m_have_datatype = true;
  }

  out_buf = m_buffer;
  out_count = m_count;
  out_datatype = m_datatype;
}

//_____________________________________________________________________________
//
void
PackBufferInfo::get_type( void        *&
                        , int          &
                        , MPI_Datatype &
                        )
{
  // Should use other overload for a PackBufferInfo
  SCI_THROW(Uintah::InternalError("get_type(void*&, int&, MPI_Datatype&) should not be called on PackBufferInfo objects", __FILE__, __LINE__));
}

//_____________________________________________________________________________
//
void
PackBufferInfo::pack( MPI_Comm comm, int & out_count )
{
  ASSERT(m_have_datatype);

  int position = 0;
  int bufsize = m_packed_buffer->getBufSize();
  //for each buffer
  for (unsigned int i = 0; i < m_start_bufs.size(); i++) {
    //pack into a contiguous buffer
    if (m_counts[i] > 0) {
      Uintah::MPI::Pack(m_start_bufs[i], m_counts[i], m_datatypes[i], m_buffer, bufsize, &position, comm);
    }
  }

  out_count = position;

  // When it is all packed, only the buffer necessarily needs to be kept around until after it is sent.
  delete m_send_list;
  m_send_list = nullptr;
  addSendlist(m_packed_buffer);
}

//_____________________________________________________________________________
//
void
PackBufferInfo::unpack( MPI_Comm comm, MPI_Status & status )
{
  ASSERT(m_have_datatype);

  unsigned long bufsize = m_packed_buffer->getBufSize();

  int position = 0;
  for (unsigned int i = 0; i < m_start_bufs.size(); i++) {
    if (m_counts[i] > 0) {
      Uintah::MPI::Unpack(m_buffer, bufsize, &position, m_start_bufs[i], m_counts[i], m_datatypes[i], comm);
    }
  }
}

