/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/Schedulers/DependencyBatch_Exp.hpp>
#include <CCA/Components/Schedulers/DetailedTasks_Exp.hpp>

namespace {

std::mutex m_lock;

}

namespace Uintah {

//______________________________________________________________________
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

//______________________________________________________________________
//
void DependencyBatch::reset()
{
  m_received         = false;
  m_made_mpi_request = false;
}

//______________________________________________________________________
//
bool DependencyBatch::makeMPIRequest()
{
  // TODO - remove lock, cleanup  conditional APH, 03/18/16
  //      - m_to_tasks may have a race condition where DependencyBatches are used in the schedulers
  if (m_to_tasks.size() > 1) {
    if (!m_made_mpi_request) {
      m_lock.lock();
      if (!m_made_mpi_request) {
        m_made_mpi_request = true;
        m_lock.unlock();
        return true;  // first to make the request
      }
      else {
        m_lock.unlock();
        return false;  // got beat out -- request already made
      }
    }
    return false;  // request already made
  }
  else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(!m_made_mpi_request);
    m_made_mpi_request = true;
    return true;
  }
}

//______________________________________________________________________
//

void DependencyBatch::received( const ProcessorGroup * pg )
{
  m_received = true;

  //set all the toVars to valid, meaning the mpi has been completed
  for (std::vector<Variable*>::iterator iter = m_to_vars.begin(); iter != m_to_vars.end(); iter++) {
    (*iter)->setValid();
  }
  for (std::list<DetailedTask*>::iterator iter = m_to_tasks.begin(); iter != m_to_tasks.end(); iter++) {
    (*iter)->decrementExternalDepCount();
    (*iter)->checkExternalDepCount();
  }

  //clear the variables that have outstanding MPI as they are completed now.
  m_to_vars.clear();
}

} // namespace Uintah
