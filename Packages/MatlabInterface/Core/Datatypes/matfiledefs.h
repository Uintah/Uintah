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
 * FILE: matfiledefs.h
 * AUTH: Jeroen G Stinstra
 * DATE: 21 FEB 2004
 */
 
#ifndef JGS_MATLABIO_MATFILEDEFS_H
#define JGS_MATLABIO_MATFILEDEFS_H 1
 
/*
 * Definitions for compiling the code
 */

// uncomment definitions to include or exclude ooptions

#define JGS_MATLABIO_USE_64INTS		1

// define 64 bit integers

#ifdef JGS_MATLABIO_USE_64INTS

#ifdef _WIN32
	typedef signed __int64 int64;
	typedef unsigned __int64 uint64;
#else
	typedef signed long long int64;
	typedef unsigned long long uint64;
#endif

#endif


#endif
