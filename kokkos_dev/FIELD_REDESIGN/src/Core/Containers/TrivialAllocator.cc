//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/TrivialAllocator.h>
#include <SCICore/Malloc/Allocator.h>
#include <stdlib.h>

namespace SCICore {
namespace Containers {

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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.4  2000/02/02 22:07:07  dmw
// Handle - added detach and Pio
// TrivialAllocator - fixed mis-allignment problem in alloc()
// Mesh - changed Nodes from LockingHandle to Handle so we won't run out
// 	of locks for semaphores when building big meshes
// Surface - just had to change the forward declaration of node
//
// Revision 1.3  1999/08/28 17:54:35  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:38:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:15  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//
