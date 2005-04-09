/*
 * Augmented edge data structure...
 * Peter-Pike Sloan	
 */

#include "AugEdge.h"

#include <Classlib/TrivialAllocator.h>

static TrivialAllocator AugEdge_alloc(sizeof(AugEdge));

void* AugEdge::operator new(size_t)
{
    return AugEdge_alloc.alloc();
}

void AugEdge::operator delete(void* rp, size_t)
{
    AugEdge_alloc.free(rp);
}
