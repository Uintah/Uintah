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
 *  LinSolver.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/LinSolver/LinSolver.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iostream.h>
#include <fstream.h>
#include "Matrix.h"

using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_LinSolver()
{
  return gov::cca::Component::pointer(new LinSolver());
}

LinSolver::LinSolver()
{
  fieldPort.setParent(this);

  int nRow=20;
  m=new Matrix(nRow,3);
  for(int r=0; r<nRow; r++){
    m->setElement(r,0,drand48());
    m->setElement(r,1,drand48());
    m->setElement(r,2,drand48());      
  }

}

LinSolver::~LinSolver()
{
  if(m!=0)delete m;
  cerr << "called ~LinSolver()\n";
}

void LinSolver::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //add provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myField2DPort::pointer fdp(&fieldPort);
  svc->addProvidesPort(fdp,"dataPort","gov.cca.Field2DPort", props);
  //c->registerUsesPort("dataPort", "Field2D",props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

gov::cca::Matrix::pointer myField2DPort::getField() 
{

  return gov::cca::Matrix::pointer(com->m);
}


 


