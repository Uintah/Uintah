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
 *  PHello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May. 2003
 *
 */

#include <sci_config.h> // For MPIPP_H on SGI
#include <CCA/Components/PHello/PHello.h>
#include <iostream>
#include <Core/CCA/PIDL/PIDL.h>
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
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  services=svc;

  //create common property for collective port
  sci::cca::TypeMap::pointer cprops= svc->createTypeMap();
  cprops->putInt("rank", mpi_rank);
  cprops->putInt("size", mpi_size);


  //add UI Port
  myUIPort::pointer uip=myUIPort::pointer(new myUIPort);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", cprops);


  //add Go Port
  myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", cprops);

  //register StringPort
  if(mpi_rank==0){
    sci::cca::TypeMap::pointer props(0); 
    svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);
  }
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
 
