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


#ifndef INTRACOMM_MPI_H
#define INTRACOMM_MPI_H 

#include <Core/CCA/Comm/Intra/IntraComm.h>

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

namespace SCIRun {
  /**************************************
  CLASS
     IntraCommMPI 
   
  DESCRIPTION
     An intra-component communication library
     intended for parallel components. Uses MPI_COMM_WORLD
     for all calls (so far).

  SEE ALSO
     IntraComm.h PIDL.h 
  ***************************************/

  class IntraCommMPI : public IntraComm {
  public:
    IntraCommMPI();
    ~IntraCommMPI();
    int send(int rank, char* bytestream, int length);
    int receive(int rank, char* bytestream, int length);
    int broadcast(int root, char* bytestream, int length);
    int async_send(int rank, char* bytestream, int length);
    int async_receive(int rank, char* bytestream, int length);
  private:
  };
}

#endif
