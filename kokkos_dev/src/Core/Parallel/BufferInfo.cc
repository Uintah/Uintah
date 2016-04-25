/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Util/Assert.h>

using namespace Uintah;
using std::vector;

BufferInfo::BufferInfo()
{
  d_have_datatype = false;
  d_free_datatype = false;
  d_sendlist      = 0;
}

BufferInfo::~BufferInfo()
{
  if( d_free_datatype ) {
    ASSERT(datatype!=MPI_DATATYPE_NULL);
    ASSERT(datatype!=MPI_INT);
    ASSERT(datatype!=MPI_DOUBLE);
    MPI::Type_free(&datatype);
    datatype = MPI_DATATYPE_NULL;
  }

  for( unsigned int i = 0; i < d_datatypes.size(); i++ ) {
    if( d_free_datatypes[i] ) {
      ASSERT( d_datatypes[i] != MPI_DATATYPE_NULL );
      ASSERT( d_datatypes[i] != MPI_INT );
      ASSERT( d_datatypes[i] != MPI_DOUBLE );
      MPI::Type_free( &d_datatypes[i] );
      d_datatypes[i] = MPI_DATATYPE_NULL;
    }
  }

  if( d_sendlist ) {
    delete d_sendlist;
    d_sendlist = 0;
  }
}

unsigned int
BufferInfo::count() const
{
  return (int)d_datatypes.size();
}

void
BufferInfo::add( void*          startbuf,
                 int            count,
                 MPI_Datatype   datatype,
                 bool           free_datatype
                 )
{
  ASSERT( !d_have_datatype );
  d_startbufs.push_back( startbuf );
  d_counts.push_back( count );
  d_datatypes.push_back( datatype );
  d_free_datatypes.push_back( free_datatype );
}

void
BufferInfo::get_type( void*&        out_buf,
                      int&          out_count,
                      MPI_Datatype& out_datatype )
{
  ASSERT(count() > 0);

  if( !d_have_datatype ) {
    if( count() == 1 ) {
      buf             = d_startbufs[0];
      cnt             = d_counts[0];
      datatype        = d_datatypes[0];
      d_free_datatype = false; // Will get freed with array
    }
    else {
      std::vector<MPI_Aint> indices(count());
      for( unsigned int i = 0; i < d_startbufs.size(); i++ ) {
        indices[i] = (MPI_Aint)d_startbufs[i];
      }
      MPI::Type_struct( count(), &d_counts[0], &indices[0], &d_datatypes[0], &datatype );
      MPI::Type_commit( &datatype );
      buf = 0;
      cnt = 1;
      d_free_datatype = true;
    }
    d_have_datatype = true;
  }
  out_buf      = buf;
  out_count    = cnt;
  out_datatype = datatype;
}

void
BufferInfo::addSendlist( RefCounted* obj )
{
  obj->addReference();
  d_sendlist = scinew Sendlist( d_sendlist, obj );
}

Sendlist*
BufferInfo::takeSendlist()
{
  Sendlist* rtn = d_sendlist;
  d_sendlist = 0; // They are now responsible for freeing...
  return rtn;
}
