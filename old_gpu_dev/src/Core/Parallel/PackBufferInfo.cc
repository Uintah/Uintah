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



#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/RefCounted.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>

using namespace Uintah;

#include <iostream>
#include <zlib.h>
#include <string.h>

PackBufferInfo::PackBufferInfo()
  : BufferInfo()
{
  packedBuffer=0;
}

PackBufferInfo::~PackBufferInfo()
{
  if (packedBuffer && packedBuffer->removeReference())
  {
    delete packedBuffer;
    packedBuffer=0;
  }
}

void
PackBufferInfo::get_type(void*& out_buf, int& out_count,
			 MPI_Datatype& out_datatype, MPI_Comm comm)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::get_type");
  ASSERT(count() > 0);
  if(!have_datatype){
    int packed_size;
    int total_packed_size=0;
    for (int i = 0; i < (int)startbufs.size(); i++) {
      if(counts[i]>0)
      {
        MPI_Pack_size(counts[i], datatypes[i], comm, &packed_size);
        total_packed_size += packed_size;
      }
    }

    packedBuffer = scinew PackedBuffer(total_packed_size);
    packedBuffer->addReference();

    datatype = MPI_PACKED;
    cnt=total_packed_size;
    buf = packedBuffer->getBuffer();
    have_datatype=true;
  }
  out_buf=buf;
  out_count=cnt;
  out_datatype=datatype;
}

void PackBufferInfo::get_type(void*&, int&, MPI_Datatype&)
{
  // Should use other overload for a PackBufferInfo
  SCI_THROW(SCIRun::InternalError("get_type(void*&, int&, MPI_Datatype&) should not be called on PackBufferInfo objects", __FILE__, __LINE__));
}


void
PackBufferInfo::pack(MPI_Comm comm, int& out_count)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::pack");
  ASSERT(have_datatype);

  int position = 0;
  int bufsize = packedBuffer->getBufSize();
  //for each buffer
  for (int i = 0; i < (int)startbufs.size(); i++) {
    //pack into a contigious buffer
    if(counts[i]>0)
      MPI_Pack(startbufs[i], counts[i], datatypes[i], buf, bufsize,
	       &position, comm);
  }
  
  out_count = position;

  // When it is all packed, only the buffer necessarily needs to be kept
  // around until after it is sent.
  delete sendlist;
  sendlist = 0;
  addSendlist(packedBuffer);
}

void
PackBufferInfo::unpack(MPI_Comm comm,MPI_Status &status)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::unpack");
  ASSERT(have_datatype);
  
  unsigned long bufsize = packedBuffer->getBufSize();

  int position = 0;
  for (int i = 0; i < (int)startbufs.size(); i++) {
    if(counts[i]>0)
    {
      MPI_Unpack(buf, bufsize, &position, startbufs[i], counts[i], datatypes[i], comm);
    }
  }
}

