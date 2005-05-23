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
 *  PWorld.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <sci_defs/config_defs.h> // For MPIPP_H on SGI
#include<mpi.h>
#include <CCA/Components/PWorld/PWorld.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/PRMI.h>
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
  PRMI::lock();
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  PRMI::unlock();
  services=svc;

  //create common property for collective port
  sci::cca::TypeMap::pointer cprops= svc->createTypeMap();
  cprops->putInt("rank", mpi_rank);
  cprops->putInt("size", mpi_size);

  //add StringPort
  StringPort::pointer strport=StringPort::pointer(new StringPort);
  svc->addProvidesPort(strport,"stringport","sci.cca.ports.StringPort",cprops); 
}

void PWorld::setCommunicator(int comm){
  //  MPI_COMM_COM=*(MPI_Comm*)(comm);
}

std::string 
StringPort::getString(){
  //int mpi_size, mpi_rank;
  //MPI_Comm_size(MPI_COMM_COM,&mpi_size);
  //MPI_Comm_rank(com->MPI_COMM_COM,&mpi_rank);
  //if(mpi_rank==0) return "PWorld One";
  //else if(mpi_rank==1) return "PWorld Two";
  //else return "PWorld ...";
  return "PWorld";
}

