#ifndef UINTAH_SCHEDULERS_COMMRECMPI_H
#define UINTAH_SCHEDULERS_COMMRECMPI_H

#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/CCA/Components/Schedulers/BatchReceiveHandler.h>

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

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
  // returns true while there are more tests to wait for.
  bool waitsome(MPI_Comm comm, int me); // return false when all done
  bool testsome(MPI_Comm comm, int me); // return false when all done

  void waitall(MPI_Comm comm, int me);
  void add(MPI_Request id, AfterCommunicationHandler* handler) {
    ids.push_back(id);
    handlers.push_back(handler);
  }

  vector<MPI_Request> ids;
  vector<AfterCommunicationHandler*> handlers;

  // temporary, used in calls to waitsome, testsome, and waitall
  vector<MPI_Status> statii;
  vector<int> indices;
private:

  bool donesome(MPI_Comm & comm, int donecount);
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

  void finishedCommunication(MPI_Comm comm)
  {
    // The order is important: it should unpack before informing the
    // DependencyBatch that the data has been received.
    if (unpackHandler_ != 0) unpackHandler_->finishedCommunication(comm);
    if (batchHandler_ != 0) batchHandler_->finishedCommunication(comm);
  }
private:
  PackBufferInfo* unpackHandler_;
  BatchReceiveHandler* batchHandler_;
};

typedef CommRecMPI<Sendlist> SendRecord;
typedef CommRecMPI<ReceiveHandler> RecvRecord;

template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
waitsome(MPI_Comm comm, int me)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  //  dbg << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Waitsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  return donesome(comm, donecount);
}

template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
testsome(MPI_Comm comm, int me)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  //  dbg << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  return donesome(comm, donecount);
}

template <class AfterCommunicationHandler>
bool CommRecMPI<AfterCommunicationHandler>::
donesome( MPI_Comm & comm, int donecount )
{
  //  dbg << me << " Done calling testsome with " << ids.size() 
  //      << " waiters and got " << donecount << " done\n";
  for(int i=0;i<donecount;i++){
    int idx=indices[i];
    if(handlers[idx]){

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "Actually received " << idx << endl;
	cerrLock.unlock();
      }
      handlers[idx]->finishedCommunication(comm);
      delete handlers[idx];
      handlers[idx]=0;
    }
    ids[idx] = MPI_REQUEST_NULL;;
  }
  if(donecount == (int)ids.size() || donecount == MPI_UNDEFINED){
    ids.clear();
    handlers.clear();
    return false; // no more to test
  }

  // removed finished requests
  int j = 0;
  for (int i=0; i < (int)ids.size(); i++) {
    if (ids[i] != MPI_REQUEST_NULL) {
      ids[j] = ids[i];
      handlers[j] = handlers[i];
      ++j;
    }
  }
  ASSERT((int)ids.size() - donecount == j);
  ids.resize(j);
  handlers.resize(j);

  return true; // more to test
}

template <class AfterCommunicationHandler>
void CommRecMPI<AfterCommunicationHandler>::
waitall(MPI_Comm comm, int me)
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
      handlers[i]->finishedCommunication(comm);
      delete handlers[i];
    }
  }
  ids.clear();
  handlers.clear();
}

}

#endif
