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
 *   March 2002
 *
 */

#include <sci_config.h> // For MPIPP_H on SGI
#include <CCA/Components/PHello/PHello.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <mpi.h>

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
  typedef char urlString[100] ;
  urlString s;

  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  services=svc;
  urlString *buf;

  sci::cca::TypeMap::pointer props = svc->createTypeMap();

  /////////////////////////////////////////////////
  //add UI Port
  myUIPort::pointer uip=myUIPort::pointer(new myUIPort);
  strcpy(s, uip->getURL().getString().c_str());
  if(mpi_rank==0){
    buf=new urlString[mpi_size];
  }
  MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
  if(mpi_rank==0){
    vector<URL> URLs; 
    for(int i=0; i<mpi_size; i++){
      string url(buf[i]);
      URLs.push_back(url);
    }
    Object::pointer obj=PIDL::objectFrom(URLs);
    sci::cca::ports::UIPort::pointer puip=pidl_cast<sci::cca::ports::UIPort::pointer>(obj);
    svc->addProvidesPort(puip,"ui","sci.cca.ports.UIPort", props);
    delete buf;
  }


  /////////////////////////////////////////////////
  //add Go Port
  myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));
  strcpy(s, gop->getURL().getString().c_str());
  if(mpi_rank==0){
    buf=new urlString[mpi_size];
  }
  MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
  if(mpi_rank==0){
    vector<URL> URLs; 
    for(int i=0; i<mpi_size; i++){
      string url(buf[i]);
      URLs.push_back(url);
    }
    Object::pointer obj=PIDL::objectFrom(URLs);
    sci::cca::ports::GoPort::pointer pgop=pidl_cast<sci::cca::ports::GoPort::pointer>(obj);
    svc->addProvidesPort(pgop,"go","sci.cca.ports.GoPort", props);
    delete buf;
  }


  /////////////////////////////////////////////////
  //register StringPort
  if(mpi_rank==0){
    svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);
  }

  MPI_Barrier(MPI_COMM_WORLD );
}

int myUIPort::ui(){
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  cout<<"UI button is clicked at #"<<mpi_rank<<"!\n";
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
  cerr<<"PHello.go.getPort...";
  
  sci::cca::Port::pointer pp=services->getPort("stringport");	
  cerr<<"Done\n";
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
 
