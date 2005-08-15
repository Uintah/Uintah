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
 *  PHello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May. 2003
 *
 */

#include <sci_defs/mpi_defs.h> // For MPIPP_H 
#include <CCA/Components/PHello/PHello.h>
#include <iostream>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/PRMI.h>
#include <mpi.h>
#include <unistd.h>
using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_PHello()
{
  return sci::cca::Component::pointer(new PHello());
}

PHello::PHello(){
}

PHello::~PHello(){
}

void PHello::setServices(const sci::cca::Services::pointer& svc){
  int mpi_size, mpi_rank;
  PRMI::lock();
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  PRMI::unlock();
  services=svc;

  //create common property for collective port

  sci::cca::TypeMap::pointer cprops= svc->createTypeMap();
  cprops->putInt("rank", mpi_rank);
  cprops->putInt("size", mpi_size);
  cprops->setRankAndSize(mpi_rank, mpi_size);
  
  sci::cca::TypeMap::pointer cprops0= svc->createTypeMap();
  cprops0->putInt("rank", mpi_rank);
  cprops0->putInt("size", mpi_size);
  //add UI Port
  myUIPort::pointer uip(new myUIPort);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", cprops0);
  
  //add Go Port
  myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", cprops);
  
  //register StringPort
  if(mpi_rank==0){
    sci::cca::TypeMap::pointer props(0); 
    svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);
  }
}

void PHello::setCommunicator(int comm){
  //  MPI_COMM_COM=*(MPI_Comm*)(comm);
}

int myUIPort::ui(){
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  cout<<"UI button is clicked at rank="<<mpi_rank<<"!\n";
  return 0;
}

myGoPort::myGoPort(const sci::cca::Services::pointer& svc){
  this->services=svc;
}

int myGoPort::go(){
  if(services.isNull()){
    cerr<<"Null services!\n";
    return 1;
  }
  sci::cca::Port::pointer pp=services->getPort("stringport");	
  if(pp.isNull()){
    cerr<<"stringport is not available!\n";
    return 1;
  }  
  sci::cca::ports::StringPort::pointer sp=pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  std::string name=sp->getString();

  services->releasePort("stringport");

  cout<<"PHello "<<name<<endl;
  return 0;
}
