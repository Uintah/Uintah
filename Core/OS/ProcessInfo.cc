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
#include <unistd.h>
#include <stdio.h>
#include <string.h>


namespace SCIRun {


  bool ProcessInfo::IsSupported ( int info_type )
  {
#if defined( __linux )
    switch ( info_type ) {
    case MEM_SIZE: return true;
    case MEM_RSS: return true;
    default: return false;
    }
    return true;
#elif defined( __sgi )
    return false;
#elif defined( __APPLE__ )
    return false;
#elif defined( _AIX )
    return false;
#else
    return false;
#endif
  }

  unsigned long ProcessInfo::GetInfo ( int info_type )
  {
#if defined( __linux )
    return LinuxGetInfo( info_type );
#elif defined( __sgi )
    return 0;
#elif defined( __APPLE__ )
    return 0;
#elif defined( _AIX )
    return 0;
#else
    return 0;
#endif
  }

  unsigned long ProcessInfo::LinuxGetInfo ( int info_type )
  {
    char statusFileName[MAXPATHLEN];
    sprintf( statusFileName, "/proc/%d/status", getpid() );

    FILE* file = fopen( statusFileName, "r" );

    if ( file != NULL ) {
      unsigned long tempLong = 0;
      char tempString[1024];
      char* compareString;

      switch ( info_type ) {
      case MEM_SIZE: compareString = "VmSize:"; break;
      case MEM_RSS:  compareString = "VmRSS:";  break;
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
  }


} // namespace SCIRun {
