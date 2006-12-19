/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  PDEdriver.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <sci_wx.h>
#include <CCA/Components/PDEdriver/PDEdriver.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <iostream>
#include <fstream>


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
  services->removeProvidesPort("go");
  services->unregisterUsesPort("pde");
  services->unregisterUsesPort("mesh");
  services->unregisterUsesPort("fem_matrix");
  services->unregisterUsesPort("linsolver");
  services->unregisterUsesPort("viewer");
}

void PDEdriver::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  //register provides ports here ...
  sci::cca::TypeMap::pointer props = svc->createTypeMap();

  PDEGoPort *gop = new PDEGoPort(svc);
  svc->addProvidesPort(PDEGoPort::pointer(gop),"go","sci.cca.ports.GoPort", props);

  svc->registerUsesPort("pde","sci.cca.ports.PDEdescriptionPort", svc->createTypeMap());
  svc->registerUsesPort("mesh","sci.cca.ports.MeshPort", svc->createTypeMap());
  svc->registerUsesPort("fem_matrix","sci.cca.ports.FEMmatrixPort", svc->createTypeMap());
  svc->registerUsesPort("linsolver","sci.cca.ports.LinSolverPort", svc->createTypeMap());
  svc->registerUsesPort("viewer","sci.cca.ports.ViewPort", svc->createTypeMap());
}

int PDEGoPort::go()
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

  sci::cca::ports::GUIService::pointer guiService;
  try {
    guiService = pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
    if (guiService.isNull()) {
      wxMessageBox(wxT("GUIService is not available"), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
      return -2;
    }
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
  }
  sci::cca::ComponentID::pointer cid = services->getComponentID();

  sci::cca::ports::PDEdescriptionPort::pointer pdePort;
  try {
    sci::cca::Port::pointer pp = services->getPort("pde");
    pdePort = pidl_cast<sci::cca::ports::PDEdescriptionPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  pdePort->getPDEdescription(nodes, boundries, dirichletNodes, dirichletValues);
  services->releasePort("pde");

  sci::cca::ports::MeshPort::pointer meshPort;
  try {
    sci::cca::Port::pointer pp = services->getPort("mesh");
    meshPort = pidl_cast<sci::cca::ports::MeshPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  meshPort->triangulate(nodes, boundries, triangles);
  services->releasePort("mesh");

  sci::cca::ports::FEMmatrixPort::pointer fem_matrixPort;
  try {
    sci::cca::Port::pointer pp = services->getPort("fem_matrix");
    fem_matrixPort = pidl_cast<sci::cca::ports::FEMmatrixPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  fem_matrixPort->makeFEMmatrices(triangles, nodes,
				  dirichletNodes, dirichletValues,
				  Ag, fg, size);
  services->releasePort("fem_matrix");

  sci::cca::ports::LinSolverPort::pointer linsolverPort;
  try {
    sci::cca::Port::pointer pp = services->getPort("linsolver");
    linsolverPort = pidl_cast<sci::cca::ports::LinSolverPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  guiService->updateProgress(cid, 30);

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
  services->releasePort("linsolver");
  guiService->updateProgress(cid, 80);

  sci::cca::ports::ViewPort::pointer viewPort;
  try {
    sci::cca::Port::pointer pp = services->getPort("viewer");
    viewPort = pidl_cast<sci::cca::ports::ViewPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("PDEdriver"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  viewPort->view2dPDE(nodes, triangles, x);
  services->releasePort("viewer");
  guiService->updateProgress(cid, 100);

  services->releasePort("cca.GUIService");

  return 0;
}

