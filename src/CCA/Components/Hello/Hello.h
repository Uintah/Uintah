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
 *  Hello.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#ifndef SCIRun_CCA_Components_Hello_h
#define SCIRun_CCA_Components_Hello_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  
  class myUIPort : public virtual sci::cca::ports::UIPort {
  public:
    int ui();
  };
  
  class myGoPort : public virtual sci::cca::ports::GoPort {
  public:
    myGoPort(const sci::cca::Services::pointer& svc);
    int go();
  private:
    sci::cca::Services::pointer services;
  };
  
  
  class Hello : public sci::cca::Component{
    
  public:
    Hello();
    ~Hello();
    void setServices(const sci::cca::Services::pointer& svc);
  private:
    
    Hello(const Hello&);
    Hello& operator=(const Hello&);
    sci::cca::Services::pointer services;
  };
  
  
} //namespace SCIRun


#endif
