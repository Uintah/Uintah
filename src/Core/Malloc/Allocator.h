/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
 *  Allocator.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#ifndef CORE_MALLOC_ALLOCATOR_H
#define CORE_MALLOC_ALLOCATOR_H

#include <sci_defs/malloc_defs.h>

#if !defined( DISABLE_SCI_MALLOC )

// need unix standard header for OS memory stuff
//  not part of standard C, so standard C++ lib doesn't include it with the other C headers
#include <unistd.h>

#include <cstdlib>



namespace Uintah {
  
struct Allocator;

extern Allocator* default_allocator;

Allocator* MakeAllocator();

void DestroyAllocator( Allocator* );

void MakeDefaultAllocator();

void PrintTag( void* );

//------------------------------------------------------------------------------
// These functions are for use in tracking down memory leaks.  In the
// MALLOC_STATS file, non-freed memory will be listed with the specified tag...
//------------------------------------------------------------------------------
const char * AllocatorSetDefaultTagMalloc( const char* tag );
const char * AllocatorSetDefaultTagNew( const char* tag );
      int    AllocatorSetDefaultTagLineNumber( int line_number );
      void   AllocatorResetDefaultTagMalloc();
      void   AllocatorResetDefaultTagNew();
      void   AllocatorResetDefaultTagLineNumber();
const char * AllocatorSetDefaultTag( const char* tag );
      void   AllocatorResetDefaultTag();


//------------------------------------------------------------------------------
// append the num to the MallocStats file if MallocStats are dumped to a file
// (negative appends nothing)
//------------------------------------------------------------------------------
void AllocatorMallocStatsAppendNumber( int num );
  
Allocator* DefaultAllocator();

void GetGlobalStats( Allocator *
                   , size_t    & nalloc
                   , size_t    & sizealloc
                   , size_t    & nfree
                   , size_t    & sizefree
                   , size_t    & nfillbin
                   , size_t    & nmmap
                   , size_t    & sizemmap
                   , size_t    & nmunmap
                   , size_t    & sizemunmap
                   , size_t    & highwater_alloc
                   , size_t    & highwater_mmap
                   , size_t    & bytes_overhead
                   , size_t    & bytes_free
                   , size_t    & bytes_fragmented
                   , size_t    & bytes_inuse
                   , size_t    & bytes_inhunks
                   );

void GetGlobalStats( Allocator *
                   , size_t    & nalloc
                   , size_t    & sizealloc
                   , size_t    & nfree
                   , size_t    & sizefree
                   , size_t    & nfillbin
                   , size_t    & nmmap
                   , size_t    & sizemmap
                   , size_t&     nmunmap
                   , size_t    & sizemunmap
                   , size_t    & highwater_alloc
                   , size_t    & highwater_mmap
                   );

int  GetNbins( Allocator* );

void GetBinStats( Allocator *
                , int         binno
                , size_t    & minsize
                , size_t    & maxsize
                , size_t    & nalloc
                , size_t    & nfree
                , size_t    & ninlist
                );

void AuditAllocator( Allocator* );

void DumpAllocator( Allocator*, const char* filename = "alloc.dump" );


//------------------------------------------------------------------------------
// Functions for locking and unlocking the allocator.
void LockAllocator( Allocator* );

void UnLockAllocator( Allocator* );
  

} // End namespace Uintah


//------------------------------------------------------------------------------
// TODO: This needs to go away, overriding global new without correspondingly overriding
//       global delete is dicey. Need to find a  way to ditch Core/Malloc altogether and
//       maintain functionality in nightly RT memory tests, APH 12/15/17
void* operator new( size_t, Uintah::Allocator*, const char*, int );
void* operator new[]( size_t, Uintah::Allocator*, const char*, int );
#define scinew new( Uintah::default_allocator, __FILE__, __LINE__ )


// SCI_MALLOC is disabled
#else

#define scinew new        // define scinew to new when defined( DISABLE_SCI_MALLOC )

#endif // !defined( DISABLE_SCI_MALLOC )

#endif // CORE_MALLOC_ALLOCATOR_H
 
