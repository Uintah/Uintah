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

#define UINTAH_USING_EXPERIMENTAL

#ifdef UINTAH_USING_EXPERIMENTAL

#include <CCA/Components/Schedulers/DependencyBatch_Exp.cpp>

#else

#include <CCA/Components/Schedulers/DependencyBatch.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>

namespace Uintah {

std::map<std::string, double> DependencyBatch::waittimes;


DependencyBatch::~DependencyBatch()
{
  DetailedDep* dep = head;
  while (dep) {
    DetailedDep* tmp = dep->next;
    delete dep;
    dep = tmp;
  }
}

void DependencyBatch::reset()
{
  received_ = false;
  madeMPIRequest_ = false;
}

bool DependencyBatch::makeMPIRequest()
{
  if (toTasks.size() > 1) {
    if (!madeMPIRequest_) {
      lock_.lock();
      if (!madeMPIRequest_) {
        madeMPIRequest_ = true;
        lock_.unlock();
        return true;  // first to make the request
      }
      else {
        lock_.unlock();
        return false;  // got beat out -- request already made
      }
    }
    return false;  // request already made
  }
  else {
    // only 1 requiring task -- don't worry about competing with another thread
    ASSERT(!madeMPIRequest_);
    madeMPIRequest_ = true;
    return true;
  }
}

void DependencyBatch::addReceiveListener( int mpiSignal )
{
  ASSERT(toTasks.size() > 1);  // only needed when multiple tasks need a batch
  lock_.lock();
  {
    receiveListeners_.insert(mpiSignal);
  }
  lock_.unlock();
}

void DependencyBatch::received( const ProcessorGroup * pg )
{
  received_ = true;

  //set all the toVars to valid, meaning the mpi has been completed
  for (std::vector<Variable*>::iterator iter = toVars.begin(); iter != toVars.end(); iter++) {
    (*iter)->setValid();
  }
  for (std::list<DetailedTask*>::iterator iter = toTasks.begin(); iter != toTasks.end(); iter++) {
    // if the count is 0, the task will add itself to the external ready queue
    //cout << pg->myrank() << "  Dec: " << *fromTask << " for " << *(*iter) << endl;
    (*iter)->decrementExternalDepCount();
    //cout << Parallel::getMPIRank() << "   task " << **(iter) << " received a message, remaining count " << (*iter)->getExternalDepCount() << endl;
    (*iter)->checkExternalDepCount();
  }

  //clear the variables that have outstanding MPI as they are completed now.
  toVars.clear();

  // TODO APH - Figure this out and clean up (01/31/15)
#if 0
  if (!receiveListeners_.empty()) {
    // only needed when multiple tasks need a batch
    ASSERT(toTasks.size() > 1);
    ASSERT(lock_ != 0);
    lock_->lock();
    {
      for (set<int>::iterator iter = receiveListeners_.begin(); iter != receiveListeners_.end(); ++iter) {
        // send WakeUp messages to threads on the same processor
        MPI::Send(0, 0, MPI_INT, pg->myrank(), *iter, pg->getComm());
      }
      receiveListeners_.clear();
    }
    lock_->unlock();
  }
#endif
}

} // namespace Uintah

#endif // UINTAH_USING_EXPERIMENTAL
