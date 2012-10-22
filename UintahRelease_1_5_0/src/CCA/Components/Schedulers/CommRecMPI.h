/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_SCHEDULERS_COMMRECMPI_H
#define UINTAH_SCHEDULERS_COMMRECMPI_H

#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Parallel/BufferInfo.h>
#include <CCA/Components/Schedulers/BatchReceiveHandler.h>

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
    : groupIDDefault_(0), totalBytes_(0) {}
  
  // returns true while there are more tests to wait for.
  //bool waitsome(MPI_Comm comm, int me); // return false when all done
  //bool testsome(MPI_Comm comm, int me); // return false when all done
  bool waitsome(const ProcessorGroup * pg, std::list<int>* finishedGroups = 0); 

  // overload waitsome to take another CommRecMPI and do a waitsome on the
  // union of the two sets
  bool waitsome(const ProcessorGroup * pg, CommRecMPI &cr,
		std::list<int>* finishedGroups = 0);
  bool testsome(const ProcessorGroup * pg, std::list<int>* finishedGroups = 0); 

  void waitall(const ProcessorGroup * pg);

  void add(MPI_Request id, int bytes, AfterCommunicationHandler* handler,
           string var, int message)
  { add(id, bytes, handler, var, message, groupIDDefault_); }
  void add(MPI_Request id, int bytes, AfterCommunicationHandler* handler,
	   string var, int message, int groupID);

  void setDefaultGroupID(int groupID)
  { groupIDDefault_ = groupID; }

  int getUnfinishedBytes() const
  { return totalBytes_; }

  unsigned long numRequests() const
  { return ids.size(); }

  void print(const ProcessorGroup * pg);

  static double WaitTimePerMessage;
private:  
  std::vector<MPI_Request> ids;
  std::vector<int> groupIDs;  
  std::vector<AfterCommunicationHandler*> handlers; 
  std::vector<int> byteCounts;
  std::vector<string> vars;
  std::vector<int> messageNums;

  std::map<int, int> groupWaitCount_; // groupID -> # receives waiting
  int groupIDDefault_;
  
  int totalBytes_;
  
  // temporary, used in calls to waitsome, testsome, and waitall
  std::vector<MPI_Status> statii;
  std::vector<int> indices;

  bool donesome(const ProcessorGroup * pg, int donecount, std::vector<MPI_Status> &statii,
		std::list<int>* finishedGroups = 0);
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
  { 
    delete unpackHandler_;
    unpackHandler_=0; 
    delete batchHandler_; 
    batchHandler_=0;
  }

  virtual void finishedCommunication(const ProcessorGroup * pg, MPI_Status &status)
  {
    // The order is important: it should unpack before informing the
    // DependencyBatch that the data has been received.
    if (unpackHandler_ != 0) unpackHandler_->finishedCommunication(pg,status);
    if (batchHandler_ != 0) batchHandler_->finishedCommunication(pg);
  }
private:
  PackBufferInfo* unpackHandler_;
  BatchReceiveHandler* batchHandler_;
};


} // end namespace Uintah

#endif
