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
 *  Hello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/PDEdriver/PDEdriver.h>
#include <iostream>
#include <fstream>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <qfiledialog.h>
#include <qmessagebox.h>


using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_PDEdriver()
{
  return sci::cca::Component::pointer(new PDEdriver());
}


PDEdriver::PDEdriver()
{
}

PDEdriver::~PDEdriver()
{
std::cerr << "PDEdriver::~PDEdriver" << std::endl;
//   services->removeProvidesPort("go");
//   services->removeProvidesPort("icon");
//   services->unregisterUsesPort("pde");
//   services->unregisterUsesPort("mesh");
//   services->unregisterUsesPort("fem_matrix");
//   services->unregisterUsesPort("linsolver");
//   services->unregisterUsesPort("viewer");
//   services->unregisterUsesPort("progress");
}

void PDEdriver::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  //register provides ports here ...  
  sci::cca::TypeMap::pointer props = svc->createTypeMap();

  myGoPort *gop = new myGoPort(svc);
  svc->addProvidesPort(myGoPort::pointer(gop),"go","sci.cca.ports.GoPort", props);
  svc->addProvidesPort(PDEComponentIcon::pointer(new PDEComponentIcon), "icon", "sci.cca.ports.ComponentIcon", svc->createTypeMap());

  svc->registerUsesPort("pde","sci.cca.ports.PDEdescriptionPort", svc->createTypeMap());
  svc->registerUsesPort("mesh","sci.cca.ports.MeshPort", svc->createTypeMap());
  svc->registerUsesPort("fem_matrix","sci.cca.ports.FEMmatrixPort", svc->createTypeMap());
  svc->registerUsesPort("linsolver","sci.cca.ports.LinSolverPort", svc->createTypeMap());
  svc->registerUsesPort("viewer","sci.cca.ports.ViewPort", svc->createTypeMap());

  svc->registerUsesPort("progress","sci.cca.ports.Progress", svc->createTypeMap());
}

myGoPort::myGoPort(const sci::cca::Services::pointer& svc) {
  this->svc = svc;
}

void myGoPort::updateProgress(int counter)
{
  if (pPtr.isNull()) return;
  pPtr->updateProgress(counter);
}

int myGoPort::go() 
{
  //driver's go() acts like a main()
  SSIDL::array1<double> nodes;
  SSIDL::array1<int> boundries; 
  SSIDL::array1<int> dirichletNodes;
  SSIDL::array1<double> dirichletValues;
  SSIDL::array1<int> triangles;
  SSIDL::array2<double> Ag;
  SSIDL::array1<double> fg;
  SSIDL::array1<double> x;

  int size = 0;
  int progCtr = 0;
  try {
    sci::cca::Port::pointer progPort = svc->getPort("progress");  
    pPtr = pidl_cast<sci::cca::ports::Progress::pointer>(progPort);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
  }  
  sci::cca::ports::PDEdescriptionPort::pointer pdePort;
  try {
    sci::cca::Port::pointer pp = svc->getPort("pde");   
    pdePort = pidl_cast<sci::cca::ports::PDEdescriptionPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
    return 1;
  }
  pdePort->getPDEdescription(nodes, boundries, dirichletNodes, dirichletValues);
  svc->releasePort("pde");
  updateProgress(++progCtr);

  sci::cca::ports::MeshPort::pointer meshPort;
  try {
    sci::cca::Port::pointer pp = svc->getPort("mesh");  
    meshPort = pidl_cast<sci::cca::ports::MeshPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
    return 1;
  }
  meshPort->triangulate(nodes, boundries, triangles);
  svc->releasePort("mesh");
  updateProgress(++progCtr);

  sci::cca::ports::FEMmatrixPort::pointer fem_matrixPort;
  try {
    sci::cca::Port::pointer pp = svc->getPort("fem_matrix");    
    fem_matrixPort = pidl_cast<sci::cca::ports::FEMmatrixPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
    return 1;
  }
  fem_matrixPort->makeFEMmatrices(triangles, nodes,
                                  dirichletNodes, dirichletValues,
                                  Ag, fg, size);
  svc->releasePort("fem_matrix");
  updateProgress(++progCtr);

  sci::cca::ports::LinSolverPort::pointer linsolverPort;
  try {
    sci::cca::Port::pointer pp = svc->getPort("linsolver"); 
    linsolverPort = pidl_cast<sci::cca::ports::LinSolverPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
    return 1;
  }
  updateProgress(++progCtr);

  /////////////////////////////
  //NEED REVERSE THIS dr[0] dr[1] AFTER KOSTA CHANGES THE
  //CONVENTION
  Index* dr[2];
  const int stride = 1;
  dr[1] = new Index(0, size, stride); //row is divided into blocks
  dr[0] = new Index(0, size, stride);  //col is not changed.
  // unused variable:
  //MxNArrayRep* arrr = new MxNArrayRep(2,dr);
  delete dr[0];
  delete dr[1]; 

  // linsolverPort->setCallerDistribution("DMatrix",arrr); 
  linsolverPort->jacobi(Ag, fg, x);
  svc->releasePort("linsolver");
  updateProgress(++progCtr);

  sci::cca::ports::ViewPort::pointer viewPort;
  try {
    sci::cca::Port::pointer pp = svc->getPort("viewer");    
    viewPort = pidl_cast<sci::cca::ports::ViewPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    QMessageBox::warning(0, "PDEdriver", e->getNote());
    return 1;
  }
  viewPort->view2dPDE(nodes, triangles, x);
  svc->releasePort("viewer");
  updateProgress(++progCtr);

  svc->releasePort("progress");
  return 0;
}


std::string PDEComponentIcon::getDisplayName()
{
  return "PDE Driver";
}

std::string PDEComponentIcon::getDescription()
{
  return "PDE Driver Component";
}

int PDEComponentIcon::getProgressBar()
{
  return 6;
}
 
std::string PDEComponentIcon::getIconShape()
{
  return "RECT";
}

