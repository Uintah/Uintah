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

#include <Core/Parallel/BufferInfo.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Util/RefCounted.h>

using namespace Uintah;


//_____________________________________________________________________________
//
BufferInfo::~BufferInfo()
{
  if (m_free_datatype) {
    ASSERT(m_datatype != MPI_DATATYPE_NULL);
    ASSERT(m_datatype != MPI_INT);
    ASSERT(m_datatype != MPI_DOUBLE);
    Uintah::MPI::Type_free(&m_datatype);
    m_datatype = MPI_DATATYPE_NULL;
  }

  for (unsigned int i = 0; i < m_datatypes.size(); i++) {
    if (m_free_datatypes[i]) {
      ASSERT(m_datatypes[i] != MPI_DATATYPE_NULL);
      ASSERT(m_datatypes[i] != MPI_INT);
      ASSERT(m_datatypes[i] != MPI_DOUBLE);
      Uintah::MPI::Type_free(&m_datatypes[i]);
      m_datatypes[i] = MPI_DATATYPE_NULL;
    }
  }

  if (m_send_list) {
    delete m_send_list;
    m_send_list = nullptr;
  }
}

//_____________________________________________________________________________
//
unsigned int
BufferInfo::count() const
{
  return (int)m_datatypes.size();
}

//_____________________________________________________________________________
//
void
BufferInfo::add( void         * startbuf
               , int            count
               , MPI_Datatype   datatype
               , bool           free_datatype
               )
{
  ASSERT( !m_have_datatype );
  m_start_bufs.push_back( startbuf );
  m_counts.push_back( count );
  m_datatypes.push_back( datatype );
  m_free_datatypes.push_back( free_datatype );
}

//_____________________________________________________________________________
//
void
BufferInfo::get_type( void        *& out_buf
                    , int          & out_count
                    , MPI_Datatype & out_datatype
                    )
{
  ASSERT(count() > 0);

  if( !m_have_datatype ) {
    if( count() == 1 ) {
      m_buffer             = m_start_bufs[0];
      m_count             = m_counts[0];
      m_datatype        = m_datatypes[0];
      m_free_datatype = false; // Will get freed with array
    }
    else {
      //MPI_Type_create_struct allows for multiple things to be sent in a single message.  
      std::vector<MPI_Aint> displacements(count());
      displacements[0] = 0; //relative to itself, should always start at zero
      for( unsigned int i = 1; i < m_start_bufs.size(); i++ ) {
        // Find how far this address is displaced from the first address.
        // From MPI's point of view, it won't know whether these offsets are from the same array or
        // from entirely different variables.  We'll take advantage of that second point.
        // It also appears to allow for negative displacements.
        displacements[i] = (MPI_Aint)((char*)m_start_bufs.at(i) - (char*)m_start_bufs.at(0));
      }
      Uintah::MPI::Type_create_struct( count(), &m_counts[0], &displacements[0], &m_datatypes[0], &m_datatype );
      Uintah::MPI::Type_commit( &m_datatype );
      m_buffer = m_start_bufs[0];
      m_count = 1;
      m_free_datatype = true;
    }
    m_have_datatype = true;
  }
  out_buf      = m_buffer;
  out_count    = m_count;
  out_datatype = m_datatype;
}

//_____________________________________________________________________________
//
Sendlist::~Sendlist()
{
  if (m_obj && m_obj->removeReference()) {
    delete m_obj;
    m_obj = nullptr;
  }

  // A little more complicated than normal, so that this doesn't need to be recursive...
  Sendlist* p = m_next;
  while( p ){

    if( p->m_obj->removeReference() ) {
      delete p->m_obj;
    }
    Sendlist* n = p->m_next;
    p->m_next = nullptr;  // So that DTOR won't recurse...
    p->m_obj  = nullptr;
    delete p;
    p = n;
  }
}

//_____________________________________________________________________________
//
void
BufferInfo::addSendlist( RefCounted * obj )
{
  obj->addReference();
  m_send_list = scinew Sendlist( m_send_list, obj );
}

//_____________________________________________________________________________
// This is a nasty contract - taker is now responsible for freeing...
Sendlist*
BufferInfo::takeSendlist()
{
  Sendlist* rtn = m_send_list;
  m_send_list = nullptr;
  return rtn;
}

