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
 *  PingPong_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef PingPong_PingPong_impl_h
#define PingPong_PingPong_impl_h

#include <testprograms/Component/pp/PingPong_sidl.h>

namespace PingPong_ns {
  class SCIRun::TypeInfo;
  class PingPong_impl : public PingPong{
  public:
    std::string pingpong(const Port::pointer &port);
    void stop();
  };

  class Port_impl : public Port{
  public:
    std::string getString(int i);
    void stop();
  };

} // End namespace pingpong

#endif

