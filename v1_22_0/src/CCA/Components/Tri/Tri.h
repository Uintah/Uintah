/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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






