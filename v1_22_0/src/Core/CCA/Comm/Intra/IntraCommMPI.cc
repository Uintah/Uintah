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


#include <Core/CCA/Comm/Intra/IntraCommMPI.h>
#include <iostream>
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
  ret = MPI_Rsend((void *)bytestream, length, MPI_BYTE, rank, 13, MPI_COMM_WORLD);

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

int IntraCommMPI::async_send(int rank, char* bytestream, int length)
{ 
  int ret;
  MPI_Request req;
  ret = MPI_Isend((void *)bytestream, length, MPI_BYTE, rank, 42, MPI_COMM_WORLD, &req);

  if(ret == MPI_SUCCESS)
    return 0;
  else
    return -1;
}

int IntraCommMPI::async_receive(int rank, char* bytestream, int length)
{
  int flag;

  MPI_Iprobe(rank,42,MPI_COMM_WORLD,&flag,MPI_STATUS_IGNORE);
  if(flag)
    MPI_Recv((void *)bytestream, length, MPI_BYTE, rank, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return flag; 
}

void IntraCommMPI::barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

int IntraCommMPI::get_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  return rank;
}

int IntraCommMPI::get_size()
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  return size;
}
