/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
#  include <sgi_stl_warnings_off.h>
#  include <unistd.h>
#  include <sgi_stl_warnings_on.h>
#endif

#include <stdlib.h>

namespace SCIRun {
  
struct Allocator;
Allocator* MakeAllocator();
void DestroyAllocator(Allocator*);

extern Allocator* default_allocator;
void MakeDefaultAllocator();

void PrintTag(void*);

// these functions are for use in tracking down memory leaks
//   in the MALLOC_STATS file, unfreed memory will be listed with
//   the specified tag
const char* AllocatorSetDefaultTagMalloc(const char* tag);
const char* AllocatorSetDefaultTagNew(const char* tag);
void AllocatorResetDefaultTagMalloc();
void AllocatorResetDefaultTagNew();
const char* AllocatorSetDefaultTag(const char* tag);
void AllocatorResetDefaultTag();

// append the num to the MallocStats file if MallocStats are dumped to a file
// (negative appends nothing)
void AllocatorMallocStatsAppendNumber(int num);
  
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

  // Functions for locking and unlocking the allocator.  In the
  // pthreads implementation, these use a recursive lock that will
  // allow the same thread to lock and unlock the allocator until
  // UnLockAllocator is called.  In other implentations this just uses
  // the regular lock and unlock functions.
  void LockAllocator(Allocator*);
  void UnLockAllocator(Allocator*);
  
} // End namespace SCIRun


#ifdef _WIN32
#define scinew new
#else
void* operator new(size_t, SCIRun::Allocator*, char*, int);
void* operator new[](size_t, SCIRun::Allocator*, char*, int);
#define scinew new(SCIRun::default_allocator, __FILE__, __LINE__)
#endif



#endif
 
