/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  TrivialAllocator.cc:  Template class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Containers/TrivialAllocator.h>
#include <Core/Malloc/Allocator.h>
#include <stdlib.h>

namespace SCIRun {

#if defined(_AIX) && defined(PAGESIZE)  // On AIX (xlC) PAGESIZE is already defined.
#  undef PAGESIZE
#endif

const int PAGESIZE = 64*1024-64;  // Leave some room for malloc's overhead

TrivialAllocator::TrivialAllocator(unsigned int _size)
: freelist(0), chunklist(0), lock("TrivialAllocator lock")
{
  int word_size=sizeof(void*);
  size=word_size*((_size+(word_size-1))/word_size);
  nalloc=(PAGESIZE-sizeof(List))/size;
  alloc_size=nalloc*size+sizeof(List);
  if(getenv("SCI_TA_DISABLE"))
    ta_disable=1;
  else
    ta_disable=0;
}

TrivialAllocator::~TrivialAllocator()
{
  for(List* p=chunklist;p!=0;){
    List* tofree=p;
    p=p->next;
    delete[] tofree;
  }
}

} // End namespace SCIRun

