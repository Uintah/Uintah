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
 *  ProcessInfo.cc:
 *
 *  Written by:
 *   Author: Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   Date: 2004/02/05
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Core/OS/ProcessInfo.h>

#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#if defined( __sgi ) || defined ( __alpha ) || defined ( _AIX )
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <sys/procfs.h>
#endif


namespace SCIRun {

  bool ProcessInfo::IsSupported ( int info_type )
  {

#if defined( __linux ) || defined( __sgi ) || defined( __alpha) || defined( _AIX )

    switch ( info_type ) {
    case MEM_SIZE: return true;
    case MEM_RSS : return true;
    default      : return false;
    }

#elif defined( __APPLE__ )
    return false;
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
      char* compareString;

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
    return 0;
#else
    return 0;
#endif

  } // unsigned long ProcessInfo::GetInfo ( int info_type )


} // namespace SCIRun {
