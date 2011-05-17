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

#include <sci_defs/malloc_defs.h>

// If ENABLE_SCI_TRACE is defined, DISABLE_SCI_MALLOC must also be defined.
// The SCI trace facility will not work if SCI Malloc is turned on.
//
// For more information on using the SCI Memory Trace facility, see Trace.h.

#if MALLOC_TRACE == 1 && ( !defined(DISABLE_SCI_MALLOC) || DISABLE_SCI_MALLOC != 1 )
#  error MALLOC_TRACE and !DISABLE_SCI_MALLOC may not both be set!
#endif

#if defined( MALLOC_TRACE )

//define define scinew to new so MallocTrace catches the calls
#  define scinew new

//include malloc trace functions
#include "MallocTrace.h"

//include problematic headers
#include <algorithm>
#include <valarray>

//turn on tracing
#include "MallocTraceOn.h"

#elif !defined( DISABLE_SCI_MALLOC )
#  ifndef _WIN32
#    include   <unistd.h>
#  endif

//set these macros to be blank so everything will compile without MallocTrace
#define MALLOC_TRACE_TAG_SCOPE(tag) ;
#define MALLOC_TRACE_TAG(tag) ;
#define MALLOC_TRACE_LOG_FILE(file) ;

#include <cstdlib>

namespace SCIRun {
  
struct Allocator;
Allocator* MakeAllocator();
void DestroyAllocator(Allocator*);

extern Allocator* default_allocator;
void MakeDefaultAllocator();

void PrintTag(void*);

//
// These functions are for use in tracking down memory leaks.  In the
// MALLOC_STATS file, non-freed memory will be listed with the specified
// tag...
//
const char* AllocatorSetDefaultTagMalloc(const char* tag);
const char* AllocatorSetDefaultTagNew(const char* tag);
int         AllocatorSetDefaultTagLineNumber(int line_number);
void        AllocatorResetDefaultTagMalloc();
void        AllocatorResetDefaultTagNew();
void        AllocatorResetDefaultTagLineNumber();
const char* AllocatorSetDefaultTag(const char* tag);
void        AllocatorResetDefaultTag();

// The MPI that comes with the Ubuntu Lenny kernel, for whatever reason, calls atexit(), which, if
// we are in a sci-malloc enabled build, causes our malloc to kick off (for the very first time), which
// causes us to call atexit(), which deadlocks and hangs.  This hack avoids that.  For most OSes, this
// should just be commented out.
#define USE_LENNY_HACK 

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
int  GetNbins(Allocator*);
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

#  ifdef _WIN32
#    define scinew new
#  else
     void* operator new(size_t, SCIRun::Allocator*, const char*, int);
     void* operator new[](size_t, SCIRun::Allocator*, const char*, int);
#    define scinew new(SCIRun::default_allocator, __FILE__, __LINE__)
#  endif

#else  // MALLOC_TRACE

   // Not tracing and not using sci malloc...
#  define scinew new

//set these macros to be blank so everything will compile without MallocTrace
#define MALLOC_TRACE_TAG_SCOPE(tag) ;
#define MALLOC_TRACE_TAG(tag) ;
#define MALLOC_TRACE_LOG_FILE(file) ;

#endif // MALLOC_TRACE

#endif // Malloc_Allocator_h 1
 
