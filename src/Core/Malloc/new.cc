/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  new.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#include <sci_defs/malloc_defs.h>

#include <Core/Malloc/mem_init.h>

#include <Core/Malloc/Allocator.h>

#ifdef MALLOC_TRACE
#  include "MallocTraceOff.h"
#endif 

#include <Core/Malloc/AllocPriv.h>
#include <new>

using namespace SCIRun;

#ifdef DISABLE_SCI_MALLOC

// These stubs are needed when your code uses these functions but
// DISABLE_SCI_MALLOC is set.
namespace SCIRun {
  const char* AllocatorSetDefaultTagNew(const char* /*tag*/) {
    return
      "AllocatorSetDefaultTagNew::NOT IMPLEMENTED.  DISABLE_SCI_MALLOC is set";
  }

  void AllocatorResetDefaultTagNew() {}

  const char* AllocatorSetDefaultTag(const char* /*tag*/) {
    return
      "AllocatorSetDefaultTag::NOT IMPLEMENTED.  DISABLE_SCI_MALLOC is set";
  }

  void AllocatorResetDefaultTag() {}
  int AllocatorSetDefaultTagLineNumber(int line_number) { return line_number; }
  void AllocatorResetDefaultTagLineNumber() {}

}
#ifndef MALLOC_TRACE
void* operator new(size_t size, Allocator*, const char*, int)
{
  void* mem=new char[size];
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void* operator new[](size_t size, Allocator*, const char*, int)
{
  void* mem=new char[size];

#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void operator delete(void* ptr, Allocator*, const char*, int)
{
  free(ptr);
}

void operator delete[](void* ptr, Allocator*, const char*, int)
{
  free(ptr);
}

#endif
#else // ifdef DISABLE_SCI_MALLOC

static const char* default_new_tag = "Unknown - operator new";
static const char* default_new_array_tag = "Unknown - operator new[]";

// the line number us an optional tag (on if configured with --enable-scinew-line-numbers)
// that can also show some information (like an interation or timestep) for each tag
int default_tag_line_number = 0;

namespace SCIRun {
  const char* AllocatorSetDefaultTagNew(const char* tag)
  {
    const char* old = default_new_tag;
    default_new_tag=tag;
    default_new_array_tag=tag;
    return old;
  }

  void AllocatorResetDefaultTagNew()
  {
    default_new_tag = "Unknown - operator new";
    default_new_array_tag = "Unknown - operator new[]";
  }

  const char* AllocatorSetDefaultTag(const char* tag)
  {
    AllocatorSetDefaultTagMalloc(tag);
    return AllocatorSetDefaultTagNew(tag);
  }

  void AllocatorResetDefaultTag()
  {
    AllocatorResetDefaultTagMalloc();
    AllocatorResetDefaultTagNew();
  }
  int AllocatorSetDefaultTagLineNumber(int line_number)
  {
    int old_num = default_tag_line_number;
    default_tag_line_number = line_number;
    return old_num;
  }
  void AllocatorResetDefaultTagLineNumber() 
  {
    default_tag_line_number = 0;
  }
}

#ifdef __sgi

// This is ugly, but necessary, since --LANG:std remaps the mangling
// of the global operator new.  This provides the "old" operator new,
// since the compiler insists on generating both.
extern "C" {
    void* __nw__GUi(size_t size)
    {
  if(!default_allocator)
      MakeDefaultAllocator();
  return default_allocator->alloc(size, "unknown - operator new", default_tag_line_number);
    }

    void* __nwa__GUi(size_t size)
    {
  if(!default_allocator)
      MakeDefaultAllocator();
  return default_allocator->alloc(size, "unknown - operator new", default_tag_line_number);
    }
}
#endif

void* operator new(size_t size) throw(std::bad_alloc)
{
  if(!default_allocator)
    MakeDefaultAllocator();
   
  void *mem=default_allocator->alloc(size, default_new_tag, default_tag_line_number);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void* operator new[](size_t size) throw(std::bad_alloc)
{
  if(!default_allocator)
    MakeDefaultAllocator();
  
  void*  mem=default_allocator->alloc(size, default_new_array_tag, default_tag_line_number);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void* operator new(size_t size, const std::nothrow_t&) throw()
{
  if(!default_allocator)
    MakeDefaultAllocator();
  void* mem=default_allocator->alloc(size, "unknown - nothrow operator new", default_tag_line_number);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void* operator new[](size_t size, const std::nothrow_t&) throw()
{
  if(!default_allocator)
    MakeDefaultAllocator();
  
  void *mem=default_allocator->alloc(size, "unknown - nothrow operator new[]", default_tag_line_number);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void operator delete(void* ptr) throw()
{
    if(!default_allocator)
  MakeDefaultAllocator();
    default_allocator->free(ptr);
}

void operator delete[](void* ptr) throw()
{
    if(!default_allocator)
  MakeDefaultAllocator();
    default_allocator->free(ptr);
}

void* operator new(size_t size, Allocator* a, const char* tag, int linenum)
{
  if(!a){
    if(!default_allocator)
      MakeDefaultAllocator();
      a=default_allocator;
  }
  void* mem=a->alloc(size, tag, linenum);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void* operator new[](size_t size, Allocator* a, const char* tag, int linenum)
{
  if(!a){
    if(!default_allocator)
      MakeDefaultAllocator();
      a=default_allocator;
  }
  void* mem=a->alloc(size, tag, linenum);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++)
    static_cast<unsigned char*>(mem)[i]=MEMORY_INIT_NUMBER;
#endif
  return mem;
}

void operator delete(void* ptr, Allocator* a, const char* tag, int linenum)
{
  if(!a){
    if(!default_allocator)
      MakeDefaultAllocator();
      a=default_allocator;
  }
  a->free(ptr);
}

void operator delete[](void* ptr, Allocator* a, const char* tag, int linenum)
{
  if(!a){
    if(!default_allocator)
      MakeDefaultAllocator();
      a=default_allocator;
  }
  a->free(ptr);
}


#ifdef MALLOC_TRACE
#  include "MallocTraceOn.h"
#endif 

#endif // ifdef DISABLE_SCI_MALLOC
