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

#ifndef _WIN32
#include <sys/param.h>
#include <unistd.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>

#if defined( __sgi ) || defined ( __alpha ) || defined ( _AIX )
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <sys/procfs.h>
#endif

#if defined( __APPLE__ )
#  include <mach/mach_init.h>
#  include <mach/task.h>
#endif


namespace SCIRun {

  bool ProcessInfo::IsSupported ( int info_type )
  {
#if defined(REDSTORM)
    return false;
#endif

#if defined( __linux ) || defined( __sgi ) || defined( __alpha) || defined( _AIX ) || defined( __APPLE__ )

    switch ( info_type ) {
    case MEM_SIZE: return true;
    case MEM_RSS : return true;
    default      : return false;
    }

#else
    return false;
#endif

  }


  unsigned long ProcessInfo::GetInfo ( int info_type )
  {

#if defined( __linux )

    char statusFileName[MAXPATHLEN];
    sprintf( statusFileName, "/proc/%d/status", getpid() );

    FILE* file = fopen( statusFileName, "r" );

    if ( file != NULL ) {
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
    
    return 0;

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

    return 0;

#elif defined( __APPLE__ )

    task_basic_info_data_t processInfo;
    mach_msg_type_number_t count;
    kern_return_t          error;

    count = TASK_BASIC_INFO_COUNT;
    error = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&processInfo, &count);

    if (error != KERN_SUCCESS) {
      return 0;
    }

    switch ( info_type ) {
    case MEM_SIZE: return processInfo.virtual_size;
    case MEM_RSS : return processInfo.resident_size;
    default:       return 0;
    }
    
    return 0;

#else
    return 0;
#endif

  } // unsigned long ProcessInfo::GetInfo ( int info_type )


} // namespace SCIRun {
