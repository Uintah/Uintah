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
 *  PLinSolver.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2003
 *
 */
#include <sci_config.h> // For MPIPP_H on SGI
#include<mpi.h>
#include <CCA/Components/PLinSolver/PLinSolver.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <qmessagebox.h>
#include <string.h>

using namespace SCIRun;
using namespace std;

extern "C" sci::cca::Component::pointer make_SCIRun_PLinSolver()
{
  return sci::cca::Component::pointer(new PLinSolver());
}

PLinSolver::PLinSolver()
{
}

PLinSolver::~PLinSolver()
{

}

void PLinSolver::setServices(const sci::cca::Services::pointer& svc)
{
  
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  services=svc;
  
  //add provides ports here ...  

  sci::cca::TypeMap::pointer cprops = svc->createTypeMap();
  cprops->putInt("rank", mpi_rank);
  cprops->putInt("size", mpi_size);

  sci::cca::TypeMap::pointer cprops1 = svc->createTypeMap();
  cprops1->putInt("rank", mpi_rank);
  cprops1->putInt("size", mpi_size);

  myGoPort *goPort=new myGoPort;
  goPort->setParent(this);
  myGoPort::pointer gop(goPort);

  fieldPort=new myField2DPort;
  fieldPort->setParent(this);
  fdp=myField2DPort::pointer(fieldPort);

  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", cprops);

  if(mpi_rank==0){
    svc->addProvidesPort(fdp,"field","sci.cca.ports.Field2DPort", sci::cca::TypeMap::pointer(0));
  }
  
  if(mpi_rank==0){
    svc->registerUsesPort("matrix", "sci.cca.ports.PDEMatrixPort",sci::cca::TypeMap::pointer(0));
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

bool PLinSolver::jacobi(const SSIDL::array2<double> &A, 
			const SSIDL::array1<double> &b, int sta, int fin)
{
  //we might set the accurracy by UI
  double eps=1e-6;
  int maxiter=1000;
  //int maxiter=10;
  
  int N=b.size();
  int M=fin-sta; // [sta, fin)

  //cerr<<"### M:N="<<M<<":"<<N<<endl;
  //cerr<<"### A.size1 size2="<<A.size1()<<":"<<A.size2()<<endl;

  double *x=new double[N];
  bzero(x, N*sizeof(double));
  for(int i=0; i<N; i++) x[i]=1.0;
  ////////////////////////////////////
  //prepare the divisions of the rows
  int *recvcounts=new int[mpi_size];
  int *displs=new int[mpi_size];
  for(int rank=0; rank<mpi_size; rank++){
    int blocksize=N/mpi_size;
    int sta=rank*blocksize;
    int fin=sta+blocksize;
    if(rank==mpi_size-1) fin=N;
    recvcounts[rank]=fin-sta;
    displs[rank]=sta;
    //cerr<<"#### recvcount displs @"<<rank<<"="<<recvcounts[rank]<<" "<<displs[rank]<<endl;
  }

  int iter;
  for(iter=0; iter<maxiter; iter++){

    /////////////////////////////////////////
    //check if accurate solution is found
    double norm=0;
    for(int I=0; I<M; I++){
      int i=sta+I;
      double res_i=0;
      for(int k=0; k<N; k++){
	res_i+=A[I][k]*x[k];
      }
      res_i-=b[i];
      norm+=res_i*res_i;
    }
    
    double norm2;
    MPI_Allreduce (&norm, &norm2, 1, 
                   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //if(mpi_rank==0)    cerr<<"iter="<<iter<<"  norm2="<<norm2<<endl;
    if(norm2<eps*eps) break;
    
    ///////////////////////////////////
    //update the solution
    for(int I=0; I<M; I++){
      int i=sta+I;
      double tmp=b[i];
      for(int k=0; k<N; k++){
	if(i==k) continue;
	tmp-=A[I][k]*x[k];
      }
      x[i]=tmp/A[I][i];
    }

    //////////////////////////////////
    //reduce x[sta]-x[fin-1] to x for every node
    //because the sending buffer and recving buffer do not overlap,
    //so we can use the same buffer: x.
    MPI_Allgatherv (x+sta, M, MPI_DOUBLE, 
		    x, recvcounts, displs, 
		    MPI_DOUBLE, MPI_COMM_WORLD);
  }

  ///////////////////////////////
  //save solution 
  solution.resize(N);
  for(int i=0; i<N; i++){
    solution[i]=x[i];
  }

  ///////////////////////////////
  //  clear up memory
  delete []x;
  delete []recvcounts;
  delete []displs;

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
    QMessageBox::warning(0, "PLinSolver", "Port matrix is not available!");
    return 1;
  }  
  sci::cca::ports::PDEMatrixPort::pointer matrixPort=
    pidl_cast<sci::cca::ports::PDEMatrixPort::pointer>(pp);

  int size=matrixPort->getSize();

  Index* dr[2];
  int blocksize=size/com->mpi_size;
  int sta=com->mpi_rank*blocksize;
  int fin=sta+blocksize;
  if(com->mpi_rank==com->mpi_size-1) fin=size;

  const int stride=1;
  /////////////////////////////
  //NEED REVERSE THIS dr[0] dr[1] AFTER KOSTA CHANGES THE
  //CONVENTION
  dr[1] = new Index(sta, fin, stride); //row is divided into blocks
  dr[0] = new Index(0, size, stride);  //col is not changed.
  MxNArrayRep* arrr = new MxNArrayRep(2,dr);
  delete dr[0];   delete dr[1]; 

  matrixPort->setCallerDistribution("DMatrix",arrr); 

  SSIDL::array2<double> A;
  matrixPort->getMatrix(A);

  SSIDL::array1<double> b=matrixPort->getVector();

  com->getServices()->releasePort("matrix");	

  //cerr<<"#### A.sizes="<<A.size1()<<":"<<A.size2()<<endl;
  if(A.size1()==0 || A.size2()==0){
    QMessageBox::warning(0, "PLinSolver", "Bad input matrix and vector!");
    return 0;
  }

  /*
  ////////////////////////////////////
  //print A for debug
  sleep(com->mpi_rank*10);
  cerr<<"MATRIX A rank="<<com->mpi_rank<<"\n======================="<<endl;
  for(int i=0;i<A.size1();i++){
    for(int j=0;j<A.size2();j++){
      cerr<<A[i][j]<<" ";
    }
    cerr<<endl;
  }
  cerr<<"====================================="<<endl;
  */

  if(!com->jacobi(A,b,sta,fin)){
    cerr<<"Jacobi method failed!"<<endl;
    return 1;
  }
  return 0;
}
  



