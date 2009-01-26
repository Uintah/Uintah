/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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




/*
 *  TrivialAllocator.h:  Template class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/rtrt/Core/TrivialAllocator.h>
#include <cstdlib>

#define PAGESIZE (256*1024-64)  // Leave some room for malloc's overhead

TrivialAllocator::TrivialAllocator(unsigned int size)
: freelist(0), chunklist(0), size(size), lock("Trivial allocator lock")
{
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

