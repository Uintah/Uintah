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
 *  Tri.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_Framework_Tri_h
#define SCIRun_Framework_Tri_h

#include <Core/CCA/spec/cca_sidl.h>
#include "Delaunay.h"

//namespace SCIRun {

#define myUIPort TriUIPort
#define myGoPort TriGoPort

class Tri;
  
class myUIPort : public virtual sci::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   virtual int ui();
   void setParent(Tri *com){this->com=com;}
   Tri *com;
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
   virtual ~myGoPort(){}
   virtual int go();
   void setParent(Tri *com){this->com=com;}
   Tri *com;
};

class myMeshPort: public virtual sci::cca::ports::MeshPort{
 public:
  virtual ~myMeshPort(){}
  virtual SSIDL::array1<int> getTriangles();
  virtual SSIDL::array1<double> getNodes();
  void setParent(Tri *com){this->com=com;}
  Tri *com;  
};

class Tri : public sci::cca::Component{
                
  public:
    Tri();
    virtual ~Tri();

    virtual void setServices(const sci::cca::Services::pointer& svc);
    Delaunay *mesh;
    sci::cca::Services::pointer getServices(){return services;}
  private:

    Tri(const Tri&);
    Tri& operator=(const Tri&);
    myUIPort uiPort;
    myGoPort goPort;
    myMeshPort meshPort;
    sci::cca::Services::pointer services;
  };
//}




#endif






