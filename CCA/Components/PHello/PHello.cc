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
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);

  services=svc;
  cerr<<"svc->createTypeMap...";
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  cerr<<"Done\n";

  myUIPort::pointer uip=myUIPort::pointer(new myUIPort);
  //myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));

  typedef char urlString[100] ;
  urlString s;
  strcpy(s, uip->getURL().getString().c_str());
  urlString *buf;
  if(mpi_rank==0){
    buf=new urlString[mpi_size];
  }

  cerr<<"Running MPI_Gather rank/size="<<mpi_rank<<"/"<<mpi_size<<endl;
  MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
  
  cerr<<"svc->addProvidesPort(uip)...";  

  //svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);

  cerr<<"Done\n";

  //  cerr<<"svc->addProvidesPort(gop)...";  
  //svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", props);
  //cerr<<"Done\n";

  if(mpi_rank==0){
    vector<URL> URLs; 
    for(int i=0; i<mpi_size; i++){
      string url(buf[i]);
      URLs.push_back(url);
      cerr<<"ui port URLs["<<i<<"]="<<url<<endl;
    }
    Object::pointer obj=PIDL::objectFrom(URLs);
    sci::cca::ports::UIPort::pointer puip0=pidl_cast<sci::cca::ports::UIPort::pointer>(obj);
    sci::cca::ports::UIPort::pointer puip=puip0;
    cerr<<"calling UIPort...\n";
    puip->ui();
    cerr<<"end calling UIPort...\n";
    //sci::cca::Port::pointer puip=pidl_cast<sci::cca::Port::pointer>(obj);

    svc->addProvidesPort(puip,"ui","sci.cca.ports.UIPort", props);
    delete buf;
  }
  MPI_Barrier(MPI_COMM_WORLD );
}

int myUIPort::ui(){
  cout<<"UI button is clicked!\n";
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
  else{
    cerr<<"stringport is not null\n";
  }
  cerr<<"PHello.go.pidl_cast...";
  sci::cca::ports::StringPort::pointer sp=pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  cerr<<"Done\n";

  cerr<<"PHello.go.port.getString...";
  std::string name=sp->getString();
  cerr<<"Done\n";


  services->releasePort("stringport");

  cout<<"PHello "<<name<<endl;
  return 0;
}
 
