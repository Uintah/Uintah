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
 *  PingThrow_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kosta Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   July 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef PingThrow_PingThrow_impl_h
#define PingThrow_PingThrow_impl_h

#include <testprograms/Component/exceptiontest/PingThrow_sidl.h>

namespace PingThrow_ns {

    class PingThrow_impl : public PingThrow {
    public:
	PingThrow_impl();
	virtual ~PingThrow_impl();
	virtual int pingthrow(int);
    };
} // End namespace pingthrow

#endif

