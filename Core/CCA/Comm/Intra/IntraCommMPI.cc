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

#include <Core/CCA/Comm/Intra/IntraCommMPI.h>

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

using namespace SCIRun;

IntraCommMPI::IntraCommMPI()
{
}

IntraCommMPI::~IntraCommMPI()
{
}

int IntraCommMPI::send(int rank, char* bytestream, int length)
{
  int ret;
  ret = MPI_Send((void *)bytestream, length, MPI_BYTE, rank, 13, MPI_COMM_WORLD);

  if(ret == MPI_SUCCESS)
    return 0;
  else
    return -1;
}

int IntraCommMPI::receive(int rank, char* bytestream, int length)
{
  int ret;

  MPI_Status status;
  ret = MPI_Recv((void *)bytestream, length, MPI_BYTE, rank, 13, 
		 MPI_COMM_WORLD, &status);
  if(ret == MPI_SUCCESS)
    return 0;
  else
    return -1;
}

int IntraCommMPI::broadcast(int root, char* bytestream, int length)
{
  int ret;
  ret = MPI_Bcast((void *)bytestream, length, MPI_BYTE, root, MPI_COMM_WORLD);   

  if(ret == MPI_SUCCESS)
    return 0;
  else
    return -1;
}

