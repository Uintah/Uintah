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
#include <Core/CCA/PIDL/PIDL.h>
#include "ListForm.h"



//namespace SCIRun {


//#define ImUIPort1 
//ImUIPort

class ZList;

  class ImUIPort1 : public virtual sci::cca::ports::UIPort {

  public:
    virtual ~ImUIPort1(){}
    virtual int ui();
    void setParent(ZList *com){this->com=com;}
  private:
    ZList *com;	   	
  };

  class ImZListPort : public virtual sci::cca::ports::ZListPort {

  public:
    ImZListPort(){};	
    virtual ~ImZListPort(){};
    virtual SSIDL::array1<double> getList();
    void setParent(ZList *com){this->com=com;}
  private:
    ZList *com;	   	
  };


class ZList : public sci::cca::Component{

  public:
    std::vector<double> datalist; 	
    ZList();
    virtual ~ZList();

    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:

    ZList(const ZList&);
    ZList& operator=(const ZList&);
    ImUIPort1 uiport;
    ImZListPort listport;
    sci::cca::Services::pointer services;
  };
//}




#endif
