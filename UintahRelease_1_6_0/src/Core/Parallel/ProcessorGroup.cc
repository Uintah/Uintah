/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Thread.h>
#include   <iostream>


using namespace Uintah;
using std::cerr;
using SCIRun::Thread;

ProcessorGroup::ProcessorGroup(const ProcessorGroup* parent,
			       MPI_Comm comm, bool allmpi,
			       int rank, int size, int threads)
   : d_parent(parent), d_rank(rank), d_size(size),  d_threads(threads),
     d_comm(comm),  d_allmpi(allmpi)
{
}

ProcessorGroup::~ProcessorGroup()
{
}

void
ProcessorGroup::setgComm(int nComm) const
{
  if (d_threads <= 1  || !d_allmpi ) return;
  int curr_size=d_gComms.size();
  if (nComm <= curr_size) return;
  d_gComms.resize(nComm);
  for (int i=curr_size; i< nComm; i++){
    if(MPI_Comm_dup(d_comm, &d_gComms[i]) != MPI_SUCCESS){
      std::cerr << "MPI Error in MPI_Comm_dup\n" ;
      Thread::exitAll(1);
    }
  }
}
