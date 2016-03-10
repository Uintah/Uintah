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

#include <CCA/Components/Schedulers/CommRecMPI.h>

#include <Core/Util/FancyAssert.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Timers/Timers.hpp>

using namespace Uintah;

namespace {

Dout g_commrec_dbg{ "CommRecMPI", false };

Timers::Simple g_wait_timer{};
Timers::Simple g_test_timer{};

}

double CommRecMPI::WaitTimePerMessage = 0;


//______________________________________________________________________
//
//______________________________________________________________________

void
CommRecMPI::add(       MPI_Request                 id,
                       int                         bytes,
                       AfterCommunicationHandler * handler,
                 const std::string               & var,
                       int                         message,
                       int                         groupID )
{
  ids_.push_back( id );
  groupIDs_.push_back( groupID );
  handlers_.push_back( handler );
  byteCounts_.push_back( bytes );
  vars_.push_back( var );
  messageNums_.push_back( message );
  totalBytes_ += bytes;

  std::map<int, int>::iterator countIter = groupWaitCount_.find( groupID );

  if (countIter == groupWaitCount_.end()) {
    groupWaitCount_[groupID] = 1;
  }
  else {
    ++(countIter->second);
  }
}

//______________________________________________________________________
//
//______________________________________________________________________

void
CommRecMPI::print( const ProcessorGroup * pg )
{
  for( unsigned int i = 0; i < ids_.size(); ++i ) {
    DOUT(true, pg->myrank() << " Message: " << byteCounts_[i] << " vars: " << " num "
              << messageNums_[i] << " Vars: " << vars_[i]);
  }
}

//______________________________________________________________________
//
//______________________________________________________________________

bool
CommRecMPI::waitsome( const ProcessorGroup * pg,
                            std::list<int> * finishedGroups /* = 0 */ )
{
  if( ids_.size() == 0 ) {
    return false; // no more to test
  }

  statii.resize( ids_.size() );
  indices.resize( ids_.size() );

  // debugging
  if (g_commrec_dbg) {
    int me = pg->myrank();
    DOUT(true, me << " Waitsome: " << ids_.size() << " waiters:");
    for (unsigned i = 0; i < messageNums_.size(); ++i) {
      DOUT(true, me << "  Num: " << messageNums_[i] << " size: " << byteCounts_[i]);
    }
  }

  int     donecount;
  g_wait_timer.reset();

  MPI_Waitsome( (int)ids_.size(), &ids_[0], &donecount, &indices[0], &statii[0] );

  double total_wait_time = g_wait_timer.seconds();
  WaitTimePerMessage = total_wait_time / donecount;

  return donesome( pg, donecount, statii, finishedGroups );
}

//______________________________________________________________________
//
//______________________________________________________________________

bool
CommRecMPI::waitsome( const ProcessorGroup * pg,
                            CommRecMPI     & cr,
                            std::list<int> * finishedGroups /* = 0 */ )
{
  int size = ids_.size() + cr.ids_.size();
  if( size == 0 ) {
    return false;
  }

  // Create a new vector for indices, as we will split it into
  // the indices for this as well as for cr.  We need to be certain
  // that *this's ids come befor cr's to split them later.
  // Since we don't really use the statii in donesome, we'll just use *this's
  statii.resize(size);

  std::vector<MPI_Request> combinedIDs;
  std::vector<MPI_Status>  mystatii;
  std::vector<MPI_Status>  crstatii;
  std::vector<int>         combinedIndices(size);
  int                      donecount;

  indices.clear();
  cr.indices.clear();

  for (unsigned int i = 0; i < ids_.size(); ++i) {
    combinedIDs.push_back(ids_[i]);
  }

  for (unsigned int i = 0; i < cr.ids_.size(); ++i) {
    combinedIDs.push_back(cr.ids_[i]);
  }

  //__________________________________
  // debugging output
  if (g_commrec_dbg) {
    int me = pg->myrank();

    DOUT(true, me << " Calling combined waitsome with " << ids_.size() << " and " << cr.ids_.size() << " waiters");
    DOUT(true, me << " Comb Waitsome: " << ids_.size() << " and " << cr.ids_.size() << " waiters:");

    for( unsigned int i = 0; i < messageNums_.size(); ++i) {
      DOUT(true, me << "  Num: " << messageNums_[i] << ", vars: " << vars_[i]);
    }

    for( unsigned int i = 0; i < cr.messageNums_.size(); ++i) {
      DOUT(true, me << "  Num: " << cr.messageNums_[i] << ", vars: " << cr.vars_[i]);
    }
  }


  g_wait_timer.reset();

  MPI_Waitsome( size, &combinedIDs[0], &donecount, &combinedIndices[0], &statii[0] );

  double total_wait_time = g_wait_timer.seconds();
  WaitTimePerMessage = total_wait_time / donecount;

  DOUT(g_commrec_dbg, "after combined waitsome");

  // now split combinedIndices and donecount into the two cr's
  int myDonecount = 0;
  int crDonecount = 0;

  int num_ids = ids_.size();
  for (int i = 0; i < donecount; ++i) {
    if (combinedIndices[i] < num_ids) {
      indices.push_back(combinedIndices[i]);
      mystatii.push_back(statii[i]);
      myDonecount++;
    } else {
      cr.indices.push_back(combinedIndices[i] - num_ids);
      crstatii.push_back(statii[i]);
      crDonecount++;
    }
  }

  // Here we want to return the donesome of '*this', as we
  // want that set to complete, but we are completing as many
  // of cr as we can

  cr.donesome( pg, crDonecount, crstatii, finishedGroups );
  return donesome( pg, myDonecount, mystatii, finishedGroups );
}

//______________________________________________________________________
//
//______________________________________________________________________

bool
CommRecMPI::testsome( const ProcessorGroup * pg,
                            std::list<int> * finishedGroups /* = 0 */ )
{
  if (ids_.size() == 0) {
    return false;  // no more to test
  }
  statii.resize(  ids_.size() );
  indices.resize( ids_.size() );

  DOUT(g_commrec_dbg, pg->myrank() << " Calling testsome with " << ids_.size() << " waiters");

  int     donecount;
  g_test_timer.reset();

  MPI_Testsome( (int)ids_.size(), &ids_[0], &donecount, &indices[0], &statii[0] );

  double total_test_time = g_test_timer.seconds();

  if (donecount > 0) {
    WaitTimePerMessage = total_test_time / donecount;
  }

  return donesome( pg, donecount,statii, finishedGroups );
}

//______________________________________________________________________
//
//______________________________________________________________________

bool
CommRecMPI::donesome( const ProcessorGroup          * pg,
                            int                       donecount,
                            std::vector<MPI_Status> & statii,
                            std::list<int>          * finishedGroups )
{
  bool anyFinished = false;
  int numReceived  = 0;
  int volReceived  = 0;

  ASSERT(donecount != MPI_UNDEFINED);

  for (int i = 0; i < donecount; ++i) {
    int idx = indices[i];

    if (handlers_[idx]) {

      DOUT(g_commrec_dbg, "Actually received " << idx);

      handlers_[idx]->finishedCommunication(pg, statii[i]);
      ASSERT(handlers_[idx] != 0);

      delete handlers_[idx];
      handlers_[idx] = 0;
    }

    numReceived++;
    volReceived += byteCounts_[idx];
    ids_[idx] = MPI_REQUEST_NULL;
    totalBytes_ -= byteCounts_[idx];
    byteCounts_[idx] = 0;
    int groupID = groupIDs_[idx];

    ASSERT(groupWaitCount_.find(groupID) != groupWaitCount_.end());

    if (--groupWaitCount_[groupID] <= 0) {
      ASSERTEQ(groupWaitCount_[groupID], 0);
      if (finishedGroups != 0) {
        finishedGroups->push_back(groupID);
      }
      anyFinished = true;
    }
  }

  if (numReceived > 0) {
    if (pg->myrank() == pg->size() / 2) {
      DOUT(g_commrec_dbg, pg->myrank() << "NumReceived: " << numReceived << " , VolReceived: " << volReceived);
    }
  }

  if (donecount == (int)ids_.size()) {
    ASSERT(totalBytes_ == 0);
    ids_.clear();
    handlers_.clear();
    byteCounts_.clear();
    groupIDs_.clear();
    vars_.clear();
    messageNums_.clear();
    return false;  // no more to test
  }

  // remove finished requests
  int j = 0;
  for (int i = 0; i < (int)ids_.size(); ++i) {
    if (ids_[i] != MPI_REQUEST_NULL) {
      ids_[j] = ids_[i];
      groupIDs_[j] = groupIDs_[i];
      handlers_[j] = handlers_[i];
      byteCounts_[j] = byteCounts_[i];
      messageNums_[j] = messageNums_[i];
      vars_[j] = vars_[i];
      ++j;
    }
  }
  ASSERT((int )ids_.size() - donecount == j);

  ids_.resize(j);
  groupIDs_.resize(j);
  handlers_.resize(j);
  vars_.resize(j);
  messageNums_.resize(j);
  byteCounts_.resize(j);

  return !anyFinished;  // keep waiting until something finished
}

//______________________________________________________________________
//
//______________________________________________________________________

void
CommRecMPI::waitall( const ProcessorGroup * pg )
{
  if (ids_.size() == 0) {
    return;
  }

  statii.resize(ids_.size());

  DOUT(g_commrec_dbg, pg->myrank() << " Calling waitall with " << ids_.size() << " waiters");

  g_wait_timer.reset();

  MPI_Waitall((int)ids_.size(), &ids_[0], &statii[0]);

  double total_wait_time = g_wait_timer.seconds();
  WaitTimePerMessage = total_wait_time / ids_.size();

  DOUT(g_commrec_dbg, pg->myrank() << " Done calling waitall with " << ids_.size() << " waiters");

  for (int i = 0; i < (int)ids_.size(); ++i) {
    if (handlers_[i]) {
      handlers_[i]->finishedCommunication(pg, statii[i]);
      ASSERT(handlers_[i] != 0);
      delete handlers_[i];
      handlers_[i] = 0;
    }
  }

  ids_.clear();
  groupIDs_.clear();
  handlers_.clear();
  byteCounts_.clear();
  messageNums_.clear();
  vars_.clear();
  totalBytes_ = 0;
}
