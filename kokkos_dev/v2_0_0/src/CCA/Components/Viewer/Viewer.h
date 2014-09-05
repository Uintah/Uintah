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
 *  Viewer.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_Viewer_h
#define SCIRun_Viewer_h

#include <Core/CCA/spec/cca_sidl.h>

//namespace SCIRun {

#define myUIPort ViewerUIPort

class Viewer;

class myUIPort : public virtual sci::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   void setParent(Viewer *com){this->com=com;}
   virtual int ui();
 private:
   Viewer *com;
};


class Viewer : public sci::cca::Component{
                
  public:
    Viewer();
    virtual ~Viewer();
    sci::cca::Services::pointer getServices(){return services;}
    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:

    Viewer(const Viewer&);
    Viewer& operator=(const Viewer&);
    myUIPort uiPort;
    sci::cca::Services::pointer services;
  };
//}




#endif
