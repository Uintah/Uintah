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



#ifndef INTRACOMM_MPI_H
#define INTRACOMM_MPI_H 

#include <Core/CCA/Comm/Intra/IntraComm.h>

#include <sci_defs/config_defs.h> // For MPIPP_H on SGI
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
    virtual ~IntraCommMPI();
    int send(int rank, char* bytestream, int length);
    int receive(int rank, char* bytestream, int length);
    int broadcast(int root, char* bytestream, int length);
    int async_send(int rank, char* bytestream, int length);
    int async_receive(int rank, char* bytestream, int length);
    int get_rank();
    int get_size();
    void barrier();
  private:
  };
}

#endif
