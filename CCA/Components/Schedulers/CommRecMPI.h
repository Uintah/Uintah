#ifndef UINTAH_SCHEDULERS_COMMRECMPI_H
#define UINTAH_SCHEDULERS_COMMRECMPI_H

#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/CCA/Components/Schedulers/BatchReceiveHandler.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>


// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern SCIRun::DebugStream mixedDebug;

namespace Uintah {

// An CommunicationRecord (CommRecMPI) keeps track of MPI_Requests and
// a AfterCommunicationHandler for each of these requests.  By calling
// testsome or waitall, it will call finishedCommunication(MPI_Comm)
// on the handlers of all finish requests and then delete these
// handlers.

template <class AfterCommunicationHandler>
class CommRecMPI {
public:
  CommRecMPI()
    : groupIDDefault_(0) {}
  
  // returns true while there are more tests to wait for.
  //bool waitsome(MPI_Comm comm, int me); // return false when all done
  //bool testsome(MPI_Comm comm, int me); // return false when all done
  bool waitsome(const ProcessorGroup * pg, list<int>* finishedGroups = 0); 
  bool testsome(const ProcessorGroup * pg, list<int>* finishedGroups = 0); 

  void waitall(const ProcessorGroup * pg);

  void add(MPI_Request id, int bytes, AfterCommunicationHandler* handler)
  { add(id, bytes, handler, groupIDDefault_); }
  void add(MPI_Request id, int bytes, AfterCommunicationHandler* handler,
	   int groupID);

  void setDefaultGroupID(int groupID)
  { groupIDDefault_ = groupID; }

  int getUnfinishedBytes() const
  { return totalBytes_; }

  unsigned long numRequests() const
  { return ids.size(); }
private:  
  vector<MPI_Request> ids;
  vector<int> groupIDs;  
  vector<AfterCommunicationHandler*> handlers; 
  vector<int> byteCounts;

  map<int, int> groupWaitCount_; // groupID -> # receives waiting
  int groupIDDefault_;
  
  int totalBytes_;
  
  // temporary, used in calls to waitsome, testsome, and waitall
  vector<MPI_Status> statii;
  vector<int> indices;

  bool donesome(const ProcessorGroup * pg, int donecount,
		list<int>* finishedGroups = 0);
};

class ReceiveHandler
{
public:
  ReceiveHandler(PackBufferInfo* unpackHandler,
		 BatchReceiveHandler* batchHandler)
    : unpackHandler_(unpackHandler), batchHandler_(batchHandler)
  {}

  ~ReceiveHandler()
  { delete unpackHandler_; delete batchHandler_; }

  void finishedCommunication(const ProcessorGroup * pg)
  {
    // The order is important: it should unpack before informing the
    // DependencyBatch that the data has been received.
    if (unpackHandler_ != 0) unpackHandler_->finishedCommunication(pg);
    if (batchHandler_ != 0) batchHandler_->finishedCommunication(pg);
  }
private:
  PackBufferInfo* unpackHandler_;
  BatchReceiveHandler* batchHandler_;
};

typedef CommRecMPI<Sendlist> SendRecord;
typedef CommRecMPI<ReceiveHandler> RecvRecord;

template <class AfterCommunicationHandler>
void CommRecMPI<AfterCommunicationHandler>::
add(MPI_Request id, int bytes, AfterCommunicationHandler* handler,
    int groupID) {
  ids.push_back(id);
  groupIDs.push_back(groupID);
  handlers.push_back(handler);
  byteCounts.push_back(bytes);
  totalBytes_ += bytes;

  map<int, int>::iterator countIter = groupWaitCount_.find(groupID);
  if (countIter == groupWaitCount_.end())
    groupWaitCount_[groupID] = 1;
  else
    ++(countIter->second);
}
  
template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
waitsome(const ProcessorGroup * pg, list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  //  dbg << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Waitsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  return donesome(pg, donecount, finishedGroups);
}

template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
testsome(const ProcessorGroup * pg, list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  //  dbg << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  return donesome(pg, donecount, finishedGroups);
}

template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
donesome( const ProcessorGroup * pg, int donecount, list<int>* finishedGroups )
{
  bool anyFinished = false;
  //  dbg << me << " Done calling testsome with " << ids.size() 
  //      << " waiters and got " << donecount << " done\n";
  ASSERT(donecount != MPI_UNDEFINED);
  for(int i=0;i<donecount;i++){
    int idx=indices[i];
    if(handlers[idx]){

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "Actually received " << idx << "\n";
	cerrLock.unlock();
      }
      handlers[idx]->finishedCommunication(pg);
      delete handlers[idx];
      handlers[idx]=0;
    }
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
  if(donecount == (int)ids.size()){
    ids.clear();
    handlers.clear();
    byteCounts.clear();
    groupIDs.clear();
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
      ++j;
    }
  }
  ASSERT((int)ids.size() - donecount == j);
  
  ids.resize(j);
  groupIDs.resize(j);
  handlers.resize(j);
  byteCounts.resize(j);

  return !anyFinished; // keep waiting until something finished
}

template <class AfterCommunicationHandler>
void CommRecMPI<AfterCommunicationHandler>::
waitall(const ProcessorGroup * pg)
{
  if(ids.size() == 0)
    return;
  statii.resize(ids.size());
  //  dbg << me << " Calling waitall with " << ids.size() << " waiters\n";
  MPI_Waitall((int)ids.size(), &ids[0], &statii[0]);
  //  dbg << me << " Done calling waitall with " 
  //      << ids.size() << " waiters\n";
  for(int i=0;i<(int)ids.size();i++){
    if(handlers[i]) {
      handlers[i]->finishedCommunication(pg);
      delete handlers[i];
    }
  }
  ids.clear();
  handlers.clear();
  byteCounts.clear();
  totalBytes_ = 0;
}

}

#endif
