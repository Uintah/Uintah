#ifndef UINTAH_SCHEDULERS_COMMRECMPI_H
#define UINTAH_SCHEDULERS_COMMRECMPI_H

#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/CCA/Components/Schedulers/BatchReceiveHandler.h>

namespace Uintah {

class ProcessorGroup;

// A CommunicationRecord (CommRecMPI) keeps track of MPI_Requests and
// an AfterCommunicationHandler for each of these requests.  By calling
// testsome, waitsome or waitall, it will call finishedCommunication(pg)
// on the handlers of all finish requests and then delete these
// handlers.

class CommRecMPI {
public:
  CommRecMPI()
    : groupIDDefault_(0) {}
  
  // returns true while there are more tests to wait for.
  //bool waitsome(MPI_Comm comm, int me); // return false when all done
  //bool testsome(MPI_Comm comm, int me); // return false when all done
  bool waitsome(const ProcessorGroup * pg, list<int>* finishedGroups = 0); 

  // overload waitsome to take another CommRecMPI and do a waitsome on the
  // union of the two sets
  bool waitsome(const ProcessorGroup * pg, CommRecMPI &cr,
		list<int>* finishedGroups = 0);
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

// AfterCommunicationHandler is defined in BufferInfo.h with Sendlist
class ReceiveHandler : public AfterCommunicationHandler
{
public:
  ReceiveHandler(PackBufferInfo* unpackHandler,
		 BatchReceiveHandler* batchHandler)
    : unpackHandler_(unpackHandler), batchHandler_(batchHandler)
  {}

  virtual ~ReceiveHandler()
  { delete unpackHandler_; delete batchHandler_; }

  virtual void finishedCommunication(const ProcessorGroup * pg)
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


} // end namespace Uintah

#endif
