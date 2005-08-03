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
 *  PingPong_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <testprograms/Component/ppArr/PingPong_impl.h>
#include <Core/Util/NotFinished.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <sci_defs/mpi_defs.h> // For MPIPP_H
#include <mpi.h>
using namespace PingPong_ns;
using namespace std;

PingPong_impl::PingPong_impl()
{
}

PingPong_impl::~PingPong_impl()
{
}

int PingPong_impl::pingpong(SSIDL::array1<string>& arg)
{
  int myrank;
  int mysize;
  MPI_Comm_size(MPI_COMM_WORLD,&mysize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  arg.resize(1);
  if(myrank==0)  arg[0]="World Rank One";
  if(myrank==1)  arg[0]="World Rank Two";
  cerr<<" parepare message:"<< arg[0]<<endl;
  return 0;
}
