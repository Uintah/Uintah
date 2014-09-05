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
 *  Tri.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/Tri/Tri.h>
#include <iostream>
#include "Delaunay.h"

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Tri()
{
  return sci::cca::Component::pointer(new Tri());
}


Tri::Tri()
{
}

Tri::~Tri()
{
  services->removeProvidesPort("mesh");
}

void Tri::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  myMeshPort::pointer meshp(new myMeshPort);
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  svc->addProvidesPort(meshp, "mesh", "sci.cca.ports.MeshPort", props);
}

int 
myMeshPort::triangulate(const SSIDL::array1<double> &nodes, const SSIDL::array1<int> &boundaries, SSIDL::array1<int> &triangles)
{
  Delaunay* mesh = new Delaunay(nodes, boundaries);
  mesh->triangulation();

  std::vector<Triangle> tri=mesh->getTriangles();
  
  for (unsigned int i=0; i<tri.size();i++) {
    triangles.push_back(tri[i].index[0] - 4);
    triangles.push_back(tri[i].index[1] - 4);
    triangles.push_back(tri[i].index[2] - 4);
  }
  delete mesh;
  return 0;
}
