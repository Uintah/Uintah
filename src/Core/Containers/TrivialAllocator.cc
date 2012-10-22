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


/*
 *  TrivialAllocator.cc:  Template class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 */

#include <Core/Containers/TrivialAllocator.h>

#include <cstdlib>

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

