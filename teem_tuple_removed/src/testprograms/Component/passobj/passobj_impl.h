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
 *  passobj_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef passobj_passobj_impl_h
#define passobj_passobj_impl_h

#include <testprograms/Component/passobj/passobj_sidl.h>

namespace passobj_ns {
  class SCIRun::TypeInfo;
  class Pass_impl : public Pass{
  public:
    Pass_impl();
    ~Pass_impl();
    std::string pass(const Port::pointer &port);
  };

  class Port_impl : public Port{
  public:
    std::string getString();
  };

} // End namespace passobj_ns

#endif

