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
 *  TableTennis.h
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#ifndef SCIRun_Framework_TableTennis_h
#define SCIRun_Framework_TableTennis_h

#include <CCA/Components/TableTennis/TableTennis_sidl.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/spec/cca_sidl.h>

//namespace SCIRun {
  
class myUIPort : public virtual sci::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   virtual int ui();
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
   virtual ~myGoPort(){}
   virtual int go();
};

class myTTPort : public virtual PP::PingPong {
public:
   virtual ~myTTPort(){}
   virtual int pingpong(int test);
};

class TableTennis : public sci::cca::Component{
                
  public:
    TableTennis();
    virtual ~TableTennis();

    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:

    TableTennis(const TableTennis&);
    TableTennis& operator=(const TableTennis&);
    myUIPort uiPort;
    myGoPort goPort;
    myTTPort ttPort;
    sci::cca::Services::pointer services;
  };
//}




#endif
