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
 *  NotFinished.h:  Consistent way to keep track of holes in the code...
 *
 *  Written by:
 *   Wayne witzel
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCI_Core_Util_SizeTypeConvert_h
#define SCI_Core_Util_SizeTypeConvert_h

#include <sci_config.h>
#if HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#if HAVE_STDINT_H
#include <stdint.h>
#endif

namespace SCIRun{

  // pass in a pointer to a 64-bit int, but depending upon nByteMode it may
  // be treated as a 32-bit int (the last half wouldn't get touched).
  unsigned long convertSizeType(uint64_t* ssize, bool swapBytes,
				int nByteMode);    

} //end namespace SCIRun
#endif
