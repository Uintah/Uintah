/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Parallel/ProcessorGroup.h>

#include "Core/Thread/Time.h"

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern SCIRun::DebugStream mixedDebug;
static DebugStream dbg("RecvTiming", false);

double CommRecMPI::WaitTimePerMessage=0;

void CommRecMPI::add(MPI_Request id, int bytes, 
		     AfterCommunicationHandler* handler, 
                     string var, int message, 
                     int groupID) {
  ids.push_back(id);
  groupIDs.push_back(groupID);
  handlers.push_back(handler);
  byteCounts.push_back(bytes);
  vars.push_back(var);
  messageNums.push_back(message);
  totalBytes_ += bytes;

  map<int, int>::iterator countIter = groupWaitCount_.find(groupID);
  if (countIter == groupWaitCount_.end())
    groupWaitCount_[groupID] = 1;
  else
    ++(countIter->second);
}

void CommRecMPI::print(const ProcessorGroup * pg)
{
  for (unsigned i = 0; i < ids.size(); i++) 
  {
    cout << pg->myrank() << " Message: " << byteCounts[i] << " vars: " << " num " << messageNums[i] << " Vars: " << vars[i] << endl;
  }
}
  
bool CommRecMPI::waitsome(const ProcessorGroup * pg, 
			  list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
 
  int me = pg->myrank();
  if (mixedDebug.active()) {
    mixedDebug << me << " Waitsome: " << ids.size() << " waiters:\n";
    for (unsigned i = 0; i < messageNums.size(); i++) {
      mixedDebug << me << "  Num: " << messageNums[i] << " size: " << byteCounts[i] << endl;
        //", vars: " 
        //    << vars[i] << endl;
    }
  }
    
  int donecount;
  clock_t start=clock();
  MPI_Waitsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  WaitTimePerMessage=(clock()-start)/(double)CLOCKS_PER_SEC/donecount;
  //mixedDebug << me <<  "after waitsome\n";
  return donesome(pg, donecount, statii, finishedGroups);
}

bool CommRecMPI::waitsome(const ProcessorGroup * pg, CommRecMPI & cr,
			  list<int>* finishedGroups /*= 0*/)
{
  int size = ids.size() + cr.ids.size();
  if (size == 0)
    return false;

  // Create a new vector for indices, as we will split it into
  // the indices for this as well as for cr.  We need to be certain 
  // that *this's ids come befor cr's to split them later.
  // Since we don't really use the statii in donesome, we'll just use *this's
  statii.resize(size);
  
  vector<MPI_Request> combinedIDs;
  vector<MPI_Status> mystatii,crstatii;
  vector<int> combinedIndices(size);
  int donecount;
  unsigned i;

  indices.clear();
  cr.indices.clear();
  
  for (i = 0; i < ids.size(); i++)
    combinedIDs.push_back(ids[i]);
  for (i = 0; i < cr.ids.size(); i++)
    combinedIDs.push_back(cr.ids[i]);

  // if (!pg->myrank())
    //cout << "Size: " << size << ", thissize: " << ids.size() 
    // << ", crsize: " << cr.ids.size() << ", combinedsize: "
    // << combinedIDs.size() << endl;

  int me = pg->myrank();
  mixedDebug << me << " Calling combined waitsome with " << ids.size()
	     << " and " << cr.ids.size() << " waiters\n";

  if (mixedDebug.active()) {
    mixedDebug << me << " Comb Waitsome: " << ids.size() << " and " 
           << cr.ids.size() << " waiters:\n";
    for (i = 0; i < messageNums.size(); i++) {
      mixedDebug << me << "  Num: " << messageNums[i] << ", vars: " 
             << vars[i] << endl;
    }
    for (i = 0; i < cr.messageNums.size(); i++) {
      mixedDebug << me << "  Num: " << cr.messageNums[i] << ", vars: " 
             << cr.vars[i] << endl;
    }
  }

  clock_t start=clock();
  MPI_Waitsome(size, &combinedIDs[0], &donecount,
	       &combinedIndices[0], &statii[0]);
  WaitTimePerMessage=(clock()-start)/(double)CLOCKS_PER_SEC/donecount;
  mixedDebug << "after combined waitsome\n";

  
  // now split combinedIndices and donecount into the two cr's
  int myDonecount = 0;
  int crDonecount = 0;
  
  int mySize = ids.size();
  for (int i = 0; i < donecount; i++) {
    if (combinedIndices[i] < mySize) {
      indices.push_back(combinedIndices[i]);
      mystatii.push_back(statii[i]);
      myDonecount++;
    } else {
      cr.indices.push_back(combinedIndices[i]-mySize);
      crstatii.push_back(statii[i]);
      crDonecount++;
    }
  }

  
  // here we want to return the donesome of *this, as we
  // want that set to complete, but we are completing as many
  // of cr as we can

  cr.donesome(pg, crDonecount,crstatii, finishedGroups);
  return donesome(pg, myDonecount,mystatii, finishedGroups);
  
}
bool CommRecMPI::testsome(const ProcessorGroup * pg, 
			  list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  int me = pg->myrank();
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << me << " Calling testsome with " << ids.size() << " waiters\n";
    cerrLock.unlock();
  }
  int donecount;
  clock_t start=clock();
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  if(donecount>0)
    WaitTimePerMessage=(clock()-start)/(double)CLOCKS_PER_SEC/donecount;
  return donesome(pg, donecount,statii, finishedGroups);
}

bool CommRecMPI::donesome( const ProcessorGroup * pg, int donecount, vector<MPI_Status>& statii,
			   list<int>* finishedGroups )
{
  bool anyFinished = false;
  int numReceived = 0;
  int volReceived = 0;
  
  //  mixedDebug << me << " Done calling testsome with " << ids.size() 
  //      << " waiters and got " << donecount << " done\n";
  ASSERT(donecount != MPI_UNDEFINED);
  for(int i=0;i<donecount;i++){
    int idx=indices[i];
    
    //if(byteCounts[idx]==112000)
    //  WAIT_FOR_DEBUGGER();

    if(handlers[idx]){

      if( mixedDebug.active() ) {
        cerrLock.lock();
        mixedDebug << "Actually received " << idx << "\n";
        cerrLock.unlock();
      }
      handlers[idx]->finishedCommunication(pg,statii[i]);
      ASSERT(handlers[idx]!=0);
      delete handlers[idx];
      handlers[idx]=0;
    }
    numReceived++;
    volReceived += byteCounts[idx];
    ids[idx] = MPI_REQUEST_NULL;
    totalBytes_ -= byteCounts[idx];
    byteCounts[idx] = 0;
    int groupID = groupIDs[idx];
    ASSERT(groupWaitCount_.find(groupID) != groupWaitCount_.end());
    if (--groupWaitCount_[groupID] <= 0) {
      ASSERTEQ(groupWaitCount_[groupID], 0);
      if (finishedGroups != 0) {
        finishedGroups->push_back(groupID);
      }
      anyFinished = true;
    }
  }
  if (dbg.active() && numReceived>0) {
    if (pg->myrank() == pg->size()/2) {
      cerrLock.lock();
      dbg << pg->myrank() << " Time: " << Time::currentSeconds() << " , NumReceived= "
         << numReceived << " , VolReceived: " << volReceived << endl;
      cerrLock.unlock();
    }
  }
  if(donecount == (int)ids.size()){
    ASSERT(totalBytes_==0);
    ids.clear();
    handlers.clear();
    byteCounts.clear();
    groupIDs.clear();
    vars.clear();
    messageNums.clear();
    return false; // no more to test
  }

  // remove finished requests
  int j = 0;
  for (int i=0; i < (int)ids.size(); i++) {
    if (ids[i] != MPI_REQUEST_NULL) {
      ids[j] = ids[i];
      groupIDs[j] = groupIDs[i];
      handlers[j] = handlers[i];
      byteCounts[j] = byteCounts[i];
      messageNums[j] = messageNums[i];
      vars[j] = vars[i];
      ++j;
    }
  }
  ASSERT((int)ids.size() - donecount == j);
  
  ids.resize(j);
  groupIDs.resize(j);
  handlers.resize(j);
  vars.resize(j);
  messageNums.resize(j);
  byteCounts.resize(j);

  return !anyFinished; // keep waiting until something finished
}

void CommRecMPI::waitall(const ProcessorGroup * pg)
{
  if(ids.size() == 0)
    return;
  statii.resize(ids.size());
  //  mixedDebug << me << " Calling waitall with " << ids.size() << " waiters\n";
  clock_t start=clock();
  MPI_Waitall((int)ids.size(), &ids[0], &statii[0]);
  WaitTimePerMessage=(clock()-start)/(double)CLOCKS_PER_SEC/ids.size();
  //  mixedDebug << me << " Done calling waitall with " 
  //      << ids.size() << " waiters\n";
  for(int i=0;i<(int)ids.size();i++){
    if(handlers[i]) {
      handlers[i]->finishedCommunication(pg,statii[i]);
      ASSERT(handlers[i]!=0);
      delete handlers[i];
      handlers[i]=0;
    }
  }
  ids.clear();
  groupIDs.clear();
  handlers.clear();
  byteCounts.clear();
  messageNums.clear();
  vars.clear();
  totalBytes_ = 0;
}
