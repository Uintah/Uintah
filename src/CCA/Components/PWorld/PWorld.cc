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
 *  PWorld.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <sci_config.h> // For MPIPP_H on SGI
#include<mpi.h>
#include <CCA/Components/PWorld/PWorld.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <iostream>
#include <sstream>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_PWorld()
{
  return sci::cca::Component::pointer(new PWorld());
}


PWorld::PWorld(){
}

PWorld::~PWorld(){
}

void PWorld::setServices(const sci::cca::Services::pointer& svc){
  typedef char urlString[100] ;
  urlString s;

  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  services=svc;
  urlString *buf;

  sci::cca::TypeMap::pointer props = svc->createTypeMap();

  /////////////////////////////////////////////////
  //add StringPort
  StringPort::pointer strport=StringPort::pointer(new StringPort);
  strcpy(s, strport->getURL().getString().c_str());
  if(mpi_rank==0){
    buf=new urlString[mpi_size];
  }
  MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
  if(mpi_rank==0){
    vector<URL> URLs; 
    for(int i=0; i<mpi_size; i++){
      string url(buf[i]);
      URLs.push_back(url);
      cerr<<"stringport URL["<<i<<"]="<<url<<endl;
    }
    Object::pointer obj=PIDL::objectFrom(URLs);
    sci::cca::ports::StringPort::pointer pstrport=pidl_cast<sci::cca::ports::StringPort::pointer>(obj);

    //cerr<<"$$$ "<<pstrport->getString()<< " $$$"<<endl ;
    
    svc->addProvidesPort(pstrport,"stringport","sci.cca.ports.StringPort", props);
    delete buf;
  }

  MPI_Barrier(MPI_COMM_WORLD );




}

std::string 
StringPort::getString(){
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  cerr<<"### mpi_rank="<<mpi_rank<<"###"<<endl;
  if(mpi_rank==0) return "PWorld One";
  else if(mpi_rank==1) return "PWorld Two";
  else return "PWorld ...";
}

