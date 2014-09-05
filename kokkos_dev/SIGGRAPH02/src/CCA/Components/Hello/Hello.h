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

#ifndef SCIRun_Framework_Hello_h
#define SCIRun_Framework_Hello_h

#include <Core/CCA/spec/cca_sidl.h>

//namespace SCIRun {
  
class myUIPort : public virtual gov::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   virtual int ui();
};

class myGoPort : public virtual gov::cca::ports::GoPort {
public:
   virtual ~myGoPort(){}
   virtual int go();
};


class Hello : public gov::cca::Component{
                
  public:
    Hello();
    virtual ~Hello();

    virtual void setServices(const gov::cca::Services::pointer& svc);
  private:

    Hello(const Hello&);
    Hello& operator=(const Hello&);
    myUIPort uiPort;
    myGoPort goPort;
    gov::cca::Services::pointer services;
  };
//}




#endif
