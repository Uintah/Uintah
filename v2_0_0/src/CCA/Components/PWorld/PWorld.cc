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

  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  services=svc;

  //create common property for collective port
  sci::cca::TypeMap::pointer cprops= svc->createTypeMap();
  cprops->putInt("rank", mpi_rank);
  cprops->putInt("size", mpi_size);

  /////////////////////////////////////////////////
  //add StringPort
  StringPort::pointer strport=StringPort::pointer(new StringPort);
  svc->addProvidesPort(strport,"stringport","sci.cca.ports.StringPort", cprops);
}

std::string 
StringPort::getString(){
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  if(mpi_rank==0) return "PWorld One";
  else if(mpi_rank==1) return "PWorld Two";
  else return "PWorld ...";
}

