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
#include <iostream>
#include <fstream>
#include <qmessagebox.h>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_LinSolver()
{
  return sci::cca::Component::pointer(new LinSolver());
}

LinSolver::LinSolver()
{
  fieldPort.setParent(this);
  goPort.setParent(this);
}

LinSolver::~LinSolver()
{

}

void LinSolver::setServices(const sci::cca::Services::pointer& svc)
{

  services=svc;
  //add provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myGoPort::pointer gop(&goPort);
  myField2DPort::pointer fdp(&fieldPort);
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", props);
  svc->addProvidesPort(fdp,"field","sci.cca.ports.Field2DPort", props);
  svc->registerUsesPort("matrix", "sci.cca.ports.PDEMatrixPort",props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

bool LinSolver::jacobi(const SSIDL::array2<double> &A, 
		       const SSIDL::array1<double> &b)
{
  //we might set the accurracy by UI
  double eps=1e-6;
  int maxiter=1000;
  
  SSIDL::array1<double> x;
  int N=b.size();
  while(x.size()<b.size()) x.push_back(1.0);

  int iter;
  
  for(iter=0; iter<maxiter; iter++){
    double norm=0;
    for(int i=0; i<N; i++){
      double res_i=0;
      for(int k=0; k<N; k++){
	res_i+=A[i][k]*x[k];
      }
      res_i-=b[i];
      norm+=res_i*res_i;
    }
    if(norm<eps*eps) break;
    //cerr<<"iter="<<iter<<"  norm2="<<norm<<endl;
    
    SSIDL::array1<double> tempx=x;
    for(int i=0; i<N; i++){
      tempx[i]=b[i];
      for(int k=0; k<N; k++){
	if(i==k) continue;
	tempx[i]-=A[i][k]*x[k];
      }
      tempx[i]/=A[i][i];
    }
    x=tempx;
  }

  solution=x;

  return iter!=maxiter;
}  

SSIDL::array1<double> myField2DPort::getField() 
{
  return com->solution;
}

int myGoPort::go() 
{
  
  sci::cca::Port::pointer pp=com->getServices()->getPort("matrix");	
  if(pp.isNull()){
    QMessageBox::warning(0, "LinSolver", "Port matrix is not available!");
    return 1;
  }  
  sci::cca::ports::PDEMatrixPort::pointer matrixPort=
    pidl_cast<sci::cca::ports::PDEMatrixPort::pointer>(pp);
  
  SSIDL::array2<double> A;
  matrixPort->getMatrix(A);
  SSIDL::array1<double> b=matrixPort->getVector();

  cerr<<"#### A.sizes="<<A.size1()<<":"<<A.size2()<<endl;
  cerr<<"#### b.sizes="<<b.size()<<endl;
 
  com->getServices()->releasePort("matrix");	
  
  if(A.size1()==0 || A.size2()==0){
    QMessageBox::warning(0, "LinSolver", "Bad input matrix and vector!");
    return 0;
  }

  //cerr<<"A.size="<<A->numOfRows()<<" x "<<A->numOfCols()<<endl;
  //cerr<<"b.size="<<b.size()<<endl;
   
  if(!com->jacobi(A,b)){
    QMessageBox::warning(0, "LinSolver", "Jacobi method failed!");
  }
  
  return 0;
}
  



