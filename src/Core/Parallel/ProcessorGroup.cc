/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <Core/Parallel/Parallel.h>

#include <iostream>

using namespace Uintah;

ProcessorGroup::ProcessorGroup( const ProcessorGroup * parent
                              ,       MPI_Comm         comm
                              ,       int              rank
                              ,       int              size
                              ,       int              threads
                              )
  : m_rank(rank)
  , m_size(size)
  , m_threads(threads)
  , m_comm(comm)
  , m_parent_group(parent)
{}

void ProcessorGroup::setGlobalComm(int num_comms) const
{
  if (m_threads <= 1) {
    return;
  }

  int curr_size = m_global_comms.size();
  if (num_comms <= curr_size) {
    return;
  }

  m_global_comms.resize(num_comms);
  for (int i = curr_size; i < num_comms; i++) {
    if (Uintah::MPI::Comm_dup(m_comm, &m_global_comms[i]) != MPI_SUCCESS) {
      std::cerr << "Rank: " << m_rank << " - MPI Error in Uintah::MPI::Comm_dup\n";
      Parallel::exitAll(1);
    }
  }
}
