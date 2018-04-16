/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Schedulers/DependencyBatch.h>

#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>

#include <sstream>


namespace Uintah {


namespace {
  Dout g_received_dbg( "DependencyBatch", "DependencyBatch", "report when a DependencyBatch is received", false );

  Uintah::MasterLock g_received_mutex{};
}


//_____________________________________________________________________________
//
DependencyBatch::DependencyBatch( int            to
                                , DetailedTask * fromTask
                                , DetailedTask * toTask
                                )
  : m_from_task(fromTask)
  , m_to_rank(to)
{
  m_to_tasks.push_back(toTask);
}


//_____________________________________________________________________________
//
DependencyBatch::~DependencyBatch()
{
  DetailedDep* dep = m_head;
  while (dep) {
    DetailedDep* tmp = dep->m_next;
    delete dep;
    dep = tmp;
  }
}

//_____________________________________________________________________________
//
void
DependencyBatch::reset()
{
  m_received = false;
  m_made_mpi_request.store( false, std::memory_order_seq_cst);
}

//_____________________________________________________________________________
//
bool
DependencyBatch::makeMPIRequest()
{
  if (m_to_tasks.size() > 1) {

    // returns true if expected compares equal to the contained value, false otherwise.
    bool expected_val = false;
    return m_made_mpi_request.compare_exchange_strong(expected_val, true);

  } else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(m_made_mpi_request.load(std::memory_order_seq_cst) == false);
    m_made_mpi_request.store(true, std::memory_order_seq_cst);
    return true;
  }
}

//_____________________________________________________________________________
//
void
DependencyBatch::received( const ProcessorGroup * pg )
{
  g_received_mutex.lock();
  {
    m_received = true;

    // set all the toVars to valid, meaning the MPI has been completed
    for (auto iter = m_to_vars.begin(); iter != m_to_vars.end(); ++iter) {
      (*iter)->setValid();
    }

    // prepare for placement into the external ready queue
    for (auto iter = m_to_tasks.begin(); iter != m_to_tasks.end(); ++iter) {
      // if the count is 0, the task will add itself to the external ready queue
      (*iter)->decrementExternalDepCount();
      (*iter)->checkExternalDepCount();
    }

    // clear the variables that have outstanding MPI as they are completed now.
    m_to_vars.clear();

    // Debug only.
    if (g_received_dbg) {
      std::ostringstream message;
      message << "Received batch message " << m_message_tag << " from task " << *m_from_task << "\n";
      for (DetailedDep* dep = m_head; dep != nullptr; dep = dep->m_next) {
        message << "\tSatisfying " << *dep << "\n";
      }
      DOUT(true, message.str());
    }
  }
  g_received_mutex.unlock();
}

//_____________________________________________________________________________
// This is called from only one point in the framework - MPIScheduler::postMPIRecvs
//  from within a mutex-protected critical section. Currently, no lock necessary
void DependencyBatch::addVar( Variable * var )
{
  m_to_vars.push_back(var);
}


} // namespace Uintah

