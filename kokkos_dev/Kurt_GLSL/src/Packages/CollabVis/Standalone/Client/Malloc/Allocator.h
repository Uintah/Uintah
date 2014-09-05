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
 *  Allocator.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Malloc_Allocator_h
#define Malloc_Allocator_h 1

#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>

namespace SCIRun {

struct Allocator;
Allocator* MakeAllocator();
void DestroyAllocator(Allocator*);

extern Allocator* default_allocator;
void MakeDefaultAllocator();

void PrintTag(void*);

  const char* AllocatorSetDefaultTagMalloc(const char* tag);
  const char* AllocatorSetDefaultTagNew(const char* tag);

Allocator* DefaultAllocator();
void GetGlobalStats(Allocator*,
		    size_t& nalloc, size_t& sizealloc,
		    size_t& nfree, size_t& sizefree,
		    size_t& nfillbin,
		    size_t& nmmap, size_t& sizemmap,
		    size_t& nmunmap, size_t& sizemunmap,
		    size_t& highwater_alloc, size_t& highwater_mmap,
		    size_t& bytes_overhead,
		    size_t& bytes_free,
		    size_t& bytes_fragmented,
		    size_t& bytes_inuse,
		    size_t& bytes_inhunks);
void GetGlobalStats(Allocator*,
		    size_t& nalloc, size_t& sizealloc,
		    size_t& nfree, size_t& sizefree,
		    size_t& nfillbin,
		    size_t& nmmap, size_t& sizemmap,
		    size_t& nmunmap, size_t& sizemunmap,
		    size_t& highwater_alloc, size_t& highwater_mmap);
int GetNbins(Allocator*);
void GetBinStats(Allocator*, int binno, size_t& minsize, size_t& maxsize,
		 size_t& nalloc, size_t& nfree, size_t& ninlist);

void AuditAllocator(Allocator*);
void DumpAllocator(Allocator*, const char* filename = "alloc.dump");

} // End namespace SCIRun


#ifdef NO_SCI_MALLOC/*_WIN32*/
#define scinew new
#else
void* operator new(size_t, SCIRun::Allocator*, char*);
void* operator new[](size_t, SCIRun::Allocator*, char*);

#define __alloc_str__(x) #x
#define __alloc_xstr__(x) __alloc_str__(x)
//#define scinew new(SCIRun::default_allocator, __FILE__)
#define scinew new(SCIRun::default_allocator, \
                   __FILE__ ":" __alloc_xstr__(__LINE__)  )
#endif



#endif
 
