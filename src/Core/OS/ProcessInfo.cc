/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
 *  ProcessInfo.cc:
 *
 *  Written by:
 *   Author: Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   Date: 2004/02/05
 *
 */

#include <Core/OS/ProcessInfo.h>

#include <sys/param.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>

#if defined( __APPLE__ )
#  include <mach/mach_init.h>
#  include <mach/task.h>
#endif

#if defined( __bgq__ )
//#  include <malloc.h>  // This include is needed for the mallinfo version (that only works for memory up to 2 GB).
#  include <spi/include/kernel/memory.h>
#endif

namespace Uintah {

bool
ProcessInfo::isSupported ( int info_type )
{

#if defined( __linux ) || defined( __APPLE__ ) || defined( __bgq__ )

  switch ( info_type ) {
    case MEM_SIZE: return true;
    case MEM_RSS : return true;
    default      : return false;
  }

#else
  return false;
#endif

}

unsigned long
ProcessInfo::getInfo( int info_type )
{
#if defined( __linux ) && !defined( __bgq__ )

  char statusFileName[MAXPATHLEN];
  sprintf( statusFileName, "/proc/%d/status", getpid() );

  FILE* file = fopen( statusFileName, "r" );

  if ( file == nullptr ) {
    // FIXME: Throw an exception!
  }
  else {
    unsigned long tempLong = 0;
    char tempString[1024];
    const char* compareString;

    switch ( info_type ) {
      case MEM_SIZE: compareString = "VmSize:"; break;
      case MEM_RSS : compareString = "VmRSS:" ; break;
      default:
        fclose( file );
        return 0;
    }

    while ( !feof( file ) ) {
      fscanf( file, "%s", tempString );
      if ( !strcmp( tempString, compareString ) ) {
        fscanf( file, "%ld", &tempLong );
        fclose( file );
        return tempLong * 1024;
      }
    }
  }

  return 0;

#elif defined( __sgi ) || defined( __alpha )

  char statusFileName[MAXPATHLEN];
  sprintf( statusFileName, "/proc/%d", getpid() );

  int file = open( statusFileName, O_RDONLY );

  if ( file != -1 ) {
    struct prpsinfo processInfo;
    if ( ioctl( file, PIOCPSINFO, &processInfo ) == -1 ) {
      close( file );
      return 0;
    }

    close( file );

    switch ( info_type ) {
      case MEM_SIZE: return processInfo.pr_size   * getpagesize();
      case MEM_RSS : return processInfo.pr_rssize * getpagesize();
      default:       return 0;
    }
  }
  else {
    // FIXME: Throw an exception!
  }
    
  return 0;

#elif defined( __bgq__ )

  // !!!NOTE!!! Not handling the difference between Resident and total
  // (like the other sections) because, in theory, BGQ only has resident
  // memory...


  uint64_t stack, heap;
  // uint64_t shared, persist, heapavail, stackavail, guard, mmap;

  // Kernel_GetMemorySize( KERNEL_MEMSIZE_SHARED,     & shared );
  // Kernel_GetMemorySize( KERNEL_MEMSIZE_PERSIST,    & persist );
  // Kernel_GetMemorySize( KERNEL_MEMSIZE_HEAPAVAIL,  & heapavail );
  // Kernel_GetMemorySize( KERNEL_MEMSIZE_STACKAVAIL, & stackavail );
  Kernel_GetMemorySize( KERNEL_MEMSIZE_STACK,      & stack );
  Kernel_GetMemorySize( KERNEL_MEMSIZE_HEAP,       & heap );
  // Kernel_GetMemorySize( KERNEL_MEMSIZE_GUARD,      & guard );
  // Kernel_GetMemorySize( KERNEL_MEMSIZE_MMAP,       & mmap );

  // printf( "Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024), (double)heapavail/(1024*1024));
  // printf( "Allocated stack: %.2f MB, avail. stack: %.2f MB\n", (double)stack/(1024*1024), (double)stackavail/(1024*1024));
  // printf( "Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n",
  //          (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));

  return heap + stack;

  // The following works (at least on Vulcan), but because mallinfo
  // uses integers, the most it will report (accurately) is 2 GB... then
  // the numbers start going negative...
  //
  //  struct mallinfo m = mallinfo();
  //  
  //  unsigned int uordblks = m.uordblks;     /* chunks in use, in bytes */
  //  unsigned int hblkhd   = m.hblkhd;       /* mmap memory in bytes */
  //  
  //  unsigned int total_heap = uordblks + hblkhd;
  //
  //  return total_heap;

#elif defined( _AIX )

  char statusFileName[MAXPATHLEN];
  sprintf( statusFileName, "/proc/%d/psinfo", getpid() );

  int file = open( statusFileName, O_RDONLY );

  if ( file != -1 ) {
    struct psinfo processInfo;
    read( file, &processInfo, sizeof( psinfo ) );

    close( file );

    switch ( info_type ) {
      case MEM_SIZE: return processInfo.pr_size   * 1024;
      case MEM_RSS : return processInfo.pr_rssize * 1024;
      default:       return 0;
    }
  }
  else {
    // FIXME: Throw an exception!
  }

  return 0;

#elif defined( __APPLE__ )

  task_basic_info_data_t processInfo;
  mach_msg_type_number_t count;
  kern_return_t          error;

  count = TASK_BASIC_INFO_COUNT;
  error = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&processInfo, &count);

  if (error != KERN_SUCCESS) {
    // FIXME: Throw an exception!
    return 0;
  }

  switch ( info_type ) {
    case MEM_SIZE: return processInfo.virtual_size;
    case MEM_RSS : return processInfo.resident_size;
    default:       return 0;
  }
    
  return 0;

#else
  #pragma error We do not know how to report memory usage on this architecture... fix me.
  return 0;
#endif

} // end getInfo()

std::string
ProcessInfo::toHumanUnits( unsigned long value )
{
  char tmp[ 64 ];
  
  sprintf( tmp, "%.2lf MBs", value / 1000000.0 );
  return tmp;
}

} // namespace Uintah
