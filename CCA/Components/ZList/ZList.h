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
 *  ZList.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#ifndef SCIRun_Framework_ZList_h
#define SCIRun_Framework_ZList_h

#include <Core/CCA/spec/cca_sidl.h>
#include "ListForm.h"
//namespace SCIRun {

  class ImUIPort : public virtual gov::cca::ports::UIPort {

  public:
    virtual ~ImUIPort(){}
    virtual void ui();
  };

  class ImGoPort : public virtual gov::cca::ports::GoPort {

  public:
    virtual ~ImGoPort(){}
    virtual int go();
  };


class ZList : public gov::cca::Component{

  public:
    ZList();
    virtual ~ZList();

    virtual void setServices(const gov::cca::Services::pointer& svc);
  private:

    ZList(const ZList&);
    ZList& operator=(const ZList&);
    ImUIPort ui, go;
    gov::cca::Services::pointer services;
  };
//}




#endif
