#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>


using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern SCIRun::DebugStream mixedDebug;


void CommRecMPI::add(MPI_Request id, int bytes, 
		     AfterCommunicationHandler* handler, int groupID) {
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
  
bool CommRecMPI::waitsome(const ProcessorGroup * pg, 
			  list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
 
  int me = pg->myrank();
  mixedDebug << me << " Calling waitsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Waitsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  mixedDebug << "after old waitsome\n";
  return donesome(pg, donecount, finishedGroups);
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
  vector<int> combinedIndices(size);
  int donecount, i;

  indices.clear();
  cr.indices.clear();
  
  for (i = 0; i < static_cast<int>(ids.size()); i++)
    combinedIDs.push_back(ids[i]);
  for (i = 0; i < static_cast<int>(cr.ids.size()); i++)
    combinedIDs.push_back(cr.ids[i]);

  // if (!pg->myrank())
    //cout << "Size: " << size << ", thissize: " << ids.size() 
    // << ", crsize: " << cr.ids.size() << ", combinedsize: "
    // << combinedIDs.size() << endl;

  int me = pg->myrank();
  mixedDebug << me << " Calling combined waitsome with " << ids.size()
	     << " and " << cr.ids.size() << " waiters\n";

  MPI_Waitsome(size, &combinedIDs[0], &donecount,
	       &combinedIndices[0], &statii[0]);
  mixedDebug << "after combined waitsome\n";

  
  // now split combinedIndices and donecount into the two cr's
  int myDonecount = 0;
  int crDonecount = 0;
  
  int mySize = ids.size();
  for (i = 0; i < donecount; i++) {
    if (combinedIndices[i] < mySize) {
      indices.push_back(combinedIndices[i]);
      myDonecount++;
    } else {
      cr.indices.push_back(combinedIndices[i]-mySize);
      crDonecount++;
    }
  }

  
  // here we want to return the donesome of *this, as we
  // want that set to complete, but we are completing as many
  // of cr as we can

  cr.donesome(pg, crDonecount, finishedGroups);
  return donesome(pg, myDonecount, finishedGroups);
  
}
bool CommRecMPI::testsome(const ProcessorGroup * pg, 
			  list<int>* finishedGroups /* = 0 */)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  int me = pg->myrank();
  mixedDebug << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  return donesome(pg, donecount, finishedGroups);
}

bool CommRecMPI::donesome( const ProcessorGroup * pg, int donecount, 
			   list<int>* finishedGroups )
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

void CommRecMPI::waitall(const ProcessorGroup * pg)
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
