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

#ifndef CORE_PARALLEL_PROCESSORGROUP_H
#define CORE_PARALLEL_PROCESSORGROUP_H

#include <Core/Parallel/UintahMPI.h>

#include <vector>

namespace Uintah {

/**************************************

 CLASS
 ProcessorGroup


 GENERAL INFORMATION

 ProcessorGroup.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
 Processor_Group


 DESCRIPTION


 WARNING

 ****************************************/

class Parallel;

class ProcessorGroup {

public:

  ~ProcessorGroup(){};

  int size() const
  {
    return m_size;
  }

  int myrank() const
  {
    return m_rank;
  }

  MPI_Comm getComm() const
  {
    return m_comm;
  }

  MPI_Comm getGlobalComm( int comm_idx ) const
  {
    if (comm_idx == -1 || m_threads <= 1) {
      return m_comm;
    } else {
      return m_global_comms[comm_idx];
    }
  }

  void setGlobalComm( int num_comms ) const;


private:

  // can only be called from Parallel
  ProcessorGroup( const ProcessorGroup * parent
                ,       MPI_Comm         comm
                ,       int              rank
                ,       int              size
                ,       int              threads
                );

  // eliminate copy, assignment and move
  ProcessorGroup( const ProcessorGroup & )            = delete;
  ProcessorGroup& operator=( const ProcessorGroup & ) = delete;
  ProcessorGroup( ProcessorGroup && )                 = delete;
  ProcessorGroup& operator=( ProcessorGroup && )      = delete;

  int  m_rank;
  int  m_size;
  int  m_threads;

  MPI_Comm                        m_comm;
  mutable std::vector<MPI_Comm>   m_global_comms;
  const ProcessorGroup          * m_parent_group;

  friend class Parallel;

};

}  // namespace Uintah

#endif // end CORE_PARALLEL_PROCESSORGROUP_H
