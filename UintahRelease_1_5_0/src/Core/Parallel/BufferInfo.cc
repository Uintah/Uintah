/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#include <Core/Util/RefCounted.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using std::vector;


BufferInfo::BufferInfo()
{
  have_datatype=false;
  free_datatype=false;
  sendlist=0;
}

BufferInfo::~BufferInfo()
{
   
  if(free_datatype)
  {
    ASSERT(datatype!=MPI_DATATYPE_NULL);
    ASSERT(datatype!=MPI_INT);
    ASSERT(datatype!=MPI_DOUBLE);
    MPI_Type_free(&datatype);
    datatype=MPI_DATATYPE_NULL;
  }
  for(int i=0;i<(int)datatypes.size();i++){
    if(free_datatypes[i])
    {
      ASSERT(datatypes[i]!=MPI_DATATYPE_NULL);
      ASSERT(datatypes[i]!=MPI_INT);
      ASSERT(datatypes[i]!=MPI_DOUBLE);
      MPI_Type_free(&datatypes[i]);
      datatypes[i]=MPI_DATATYPE_NULL;
    }
  }

  if(sendlist)
  {
    delete sendlist;
    sendlist=0;
  }
}

int
BufferInfo::count() const
{
  return (int)datatypes.size();
}

void
BufferInfo::add(void* startbuf, int count, MPI_Datatype datatype,
		bool free_datatype)
{
  ASSERT(!have_datatype);
  startbufs.push_back(startbuf);
  counts.push_back(count);
  datatypes.push_back(datatype);
  free_datatypes.push_back(free_datatype);
} 

void
BufferInfo::get_type(void*& out_buf, int& out_count,
		     MPI_Datatype& out_datatype)
{
  ASSERT(count() > 0);
  if(!have_datatype){
    if(count() == 1){
      buf=startbufs[0];
      cnt=counts[0];
      datatype=datatypes[0];
      free_datatype=false; // Will get freed with array
    } else {
      std::vector<MPI_Aint> indices(count());
      for(int i=0;i<(int)startbufs.size();i++)
	      indices[i]=(MPI_Aint)startbufs[i];
      MPI_Type_struct(count(), &counts[0], &indices[0], &datatypes[0],
		      &datatype);
      MPI_Type_commit(&datatype);
      buf=0;
      cnt=1;
      free_datatype=true;
    }
    have_datatype=true;
  }
  out_buf=buf;
  out_count=cnt;
  out_datatype=datatype;
}

Sendlist::~Sendlist()
{
  if(obj && obj->removeReference())
  {
    delete obj;
    obj=0;
  }

  // A little more complicated than normal, so that this doesn't need
  // to be recursive...
  Sendlist* p = next;
  while(p){
    if(p->obj->removeReference())
      delete p->obj;
    Sendlist* n = p->next;
    p->next=0;  // So that DTOR won't recurse...
    p->obj=0;
    delete p;
    p=n;
  }
}

void BufferInfo::addSendlist(RefCounted* obj)
{
  obj->addReference();
  sendlist=scinew Sendlist(sendlist, obj);
}

Sendlist* BufferInfo::takeSendlist()
{
  Sendlist* rtn = sendlist;
  sendlist = 0; // They are now responsible for freeing...
  return rtn;
}
