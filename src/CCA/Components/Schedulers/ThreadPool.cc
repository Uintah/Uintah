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


#include <CCA/Components/Schedulers/ThreadPool.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/MixedScheduler.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/SendState.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/Handle.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Time.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <sstream>
#include <cstdio>


using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;

#define DAV_DEBUG 1

#ifdef USE_TAU_PROFILING
// Turn this on to get profiling split up by patches
// #define TAU_SEPARATE_PATCHES

map<string,int> taskname_to_id_map;
int unique_id = 9999;

int
create_tau_mapping( const string & taskname, const PatchSubset * patches )
{
  string full_name = taskname;

#ifdef TAU_SEPARATE_PATCHES 
  if( patches ) {
    for(int i=0;i<patches->size();i++) {

      ostringstream patch_num;
      patch_num << patches->get(i)->getID();

      full_name = full_name + "-" + patch_num.str();
    }
  }
#else  //TAU_SEPARATE_PATCHES
  int levelnum = 0;
  if( patches && patches->size() > 0) {
    levelnum = patches->get(0)->getLevel()->getIndex();
  }
  ostringstream level;
  level << "Level " << levelnum;

  full_name = full_name + "-" + level.str();
#endif //TAU_SEPARATE_PATCHES

  map<string,int>::iterator iter = taskname_to_id_map.find( full_name );
  if( iter != taskname_to_id_map.end() )
    {
      return (*iter).second;
    }
  else
    {
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "creating TAU mapping for: " << full_name << "\n";
	cerrLock.unlock();
      }
      TAU_MAPPING_CREATE( full_name, "[MPIScheduler::execute()]",
			  (TauGroup_t) unique_id, full_name.c_str(), 0 );
      taskname_to_id_map[ full_name ] = unique_id;
      unique_id++;
      return (unique_id - 1);
    }
}
#endif // USE_TAU_PROFILING

/////////////// Worker //////////////////

Worker::Worker( ThreadPool * parent, int id, 
		Mutex * ready ) : 
  d_ready( ready ), d_id( id ), d_parent( parent ), d_task( 0 ), quit_( false )
{
  proc_group_ = Parallel::getRootProcessorGroup()->myrank();
}

void
Worker::quit()
{
  quit_ = true;
  d_ready->unlock();  
}

void
Worker::assignTask( DetailedTask          * task)
{
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << proc_group_ << "-" << d_id 
	       << ": assignTask:   " << *task << "\n";
    cerrLock.unlock();
  }
#endif
  ASSERT( !d_task );

  d_task = task;
}

void
Worker::run()
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("Worker_run()", "void ()", TAU_DEFAULT);
  TAU_PROFILE_TIMER(doittimer, "doit Task", "[Worker::run()]", TAU_DEFAULT);

  //if( mixedDebug.active() ) {
   cerrLock.lock();
   cerr/*mixedDebug*/ << "Worker " << proc_group_ << "-" << d_id 
	       << ", Begin run() -- PID is " << getpid() << "\n";
   cerrLock.unlock();
    //}

  for(;;){

    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " waiting for a task\n";
      cerrLock.unlock();
    }

    d_ready->lock();

    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " unlocked\n";
      cerrLock.unlock();
    }

    if( quit_ ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " quitting\n";
      cerrLock.unlock();
      TAU_PROFILE_EXIT( "thread quitting" );
      return;
    }

    double beginTime = Time::currentSeconds();
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker "  << proc_group_ << "-" << d_id 
				  << " running: " << *d_task << "\n";
      cerrLock.unlock();
    }
#endif

    ASSERT( d_task != 0 );

    DetailedTask * task = d_task;

#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
				  << " calling doit for detailed task: " 
				  << task << " which has task: " 
				  << task->getTask() << "\n";
      cerrLock.unlock();
    }
#endif
    
    try {
      d_parent->getScheduler()->runTask(task, 0); // calls task->doit - TODO: Fix iteration when we do MixedScheduler again
    } catch (Exception& e) {

      cerrLock.lock(); cerr << "Worker " << proc_group_ << "-" << d_id 
			    << ": Caught exception: " 
			    << e.message() << '\n';
      if(e.stackTrace())
	cerr << "Stack trace: " << e.stackTrace() << '\n';
      cerrLock.unlock();

      //      Thread::exitAll(1);
    } catch (std::bad_alloc e) {
      cerrLock.lock();
      cerr << "Worker " << proc_group_ << "-" << d_id 
	   << ": Caught std exception 'std::bad_alloc': " << e.what() << '\n';
      cerrLock.unlock();
      //      Thread::exitAll(1);
    } catch (std::bad_exception e) {
      cerrLock.lock();
      cerr << "Worker " << proc_group_ << "-" << d_id 
	   << ": Caught std exception 'std::bad_exception: " << e.what() << '\n';
      cerrLock.unlock();
      //      Thread::exitAll(1);
    } catch (std::ios_base::failure e) {
      cerrLock.lock();
      cerr << "Worker " << proc_group_ << "-" << d_id 
	   << ": Caught std exception 'std::ios_base::failure': " << e.what() << '\n';
      cerrLock.unlock();
      //      Thread::exitAll(1);
    } catch (std::exception e){
      cerrLock.lock();
      cerr << "Worker " << proc_group_ << "-" << d_id 
	   << ": Caught std exception: " << e.what() << '\n';
      cerrLock.unlock();
      //      Thread::exitAll(1);
    } catch(...){
      cerrLock.lock(); cerr << "Worker " << proc_group_ << "-" << d_id 
			    << ": Caught unknown exception\n";
      cerrLock.unlock();
      //      Thread::exitAll(1);
    }

#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock(); mixedDebug << "Worker " << proc_group_ << "-" << d_id 
			    << " done with done()\n";
      cerrLock.unlock();
    }
#endif

    d_task       = 0;

    double endTime = Time::currentSeconds();

    d_parent->done( d_id, task, endTime - beginTime );
#if DAV_DEBUG
    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << " told parent i am done\n";
      cerrLock.unlock();
    }
#endif
  }
}


/////////////// Receiver //////////////////

Receiver::Receiver( ThreadPool * parent, int id )
  : d_id(id), d_parent(parent), pg_(0),
    d_lock("ThreadPool Receiver lock"),
    quit_(false)
{
  proc_group_ = Parallel::getRootProcessorGroup()->myrank();  
}

void
Receiver::assignTask(DetailedTask* task)
{
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Receiver " << proc_group_ << "-" << d_id 
	       << ": assignTask:   " << *task << "\n";
    cerrLock.unlock();
  }
#endif
 d_lock.lock();
  if (pg_ == 0 /* signals that it is blocked waiting for any task */) {
    pg_ = d_parent->getScheduler()->getProcessorGroup();
    cerr << "Resuming thread " << my_thread_ << "\n";
    my_thread_->resume();
  }
  newTasks_.push(task);
 d_lock.unlock();
  MPI_Send(0, 0, MPI_INT, pg_->myrank(), d_id, pg_->getComm());  
}

void
Receiver::quit()
{
  ASSERT(pg_ == 0); // must be finished with its work before calling this
  quit_ = true;
  my_thread_->resume();  
}

bool
Receiver::addAwaitingTasks()
{
  d_lock.lock();

  bool addedTasks = false;

  // move newTasks to awaitingTasks
  while (!newTasks_.empty()) {
    DetailedTask* task = newTasks_.front();
    newTasks_.pop();
    
    // If a slot in the awaitingTasks_ array is available, use it;
    // otherwise make the array bigger.
    int slot = -1;
    if (!availTaskSlots_.empty()) {
      slot = availTaskSlots_.front();
      availTaskSlots_.pop();
    }
    else {
      slot = (int)awaitingTasks_.size();
      awaitingTasks_.push_back(0);
    }

    recvs_.setDefaultGroupID(slot);    
    list<DependencyBatch*> externalRecvs;
    unsigned long prevNumRequests = recvs_.numRequests();
    d_parent->getScheduler()->postMPIRecvs(task, false, 0, 0); // FIX iteration
    ASSERT(awaitingTasks_[slot] == 0);
    awaitingTasks_[slot] = scinew AwaitingTask(task, externalRecvs, d_id);
    if (recvs_.numRequests() == prevNumRequests) {
      // No new requests added -- so the task is not waiting on any receives
      // it has personally posted.
      semiReadyTasks_.push_back(slot);
    }
    addedTasks = true;
  }
  d_lock.unlock();
  
  return addedTasks;
}

void
Receiver::run()
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("Receiver_run()", "void ()", TAU_DEFAULT);
  TAU_PROFILE_TIMER(doittimer, "doit Task", "[Receiver::run()]", TAU_DEFAULT);

  cerr << "Running thread " << my_thread_ << "\n";

  //if( mixedDebug.active() ) {
    cerrLock.lock();
    cerr/*mixedDebug*/ << "Receiver " << proc_group_ << "-" << d_id 
	       << ", Begin run() -- PID is " << getpid() << "\n";
    cerrLock.unlock();
    //}

  MPI_Request wakeUpRequest;
  MPI_Status status;  

  cout << "Here: pg_ is " << pg_ << "\n";
  cout << " and size of it is: " << pg_->size() << "\n";
  cout << " and rank of it is: " << pg_->myrank() << "\n";

  // post receive for self-mpi-process wake up signal
  MPI_Irecv(0, 0, MPI_INT, pg_->myrank(), d_id, pg_->getComm(),
	    &wakeUpRequest);
  recvs_.add(wakeUpRequest, 0, 0,"",0,-1);

  int preBytesLeft = recvs_.getUnfinishedBytes();

  ASSERT(pg_ != 0);  
  for(;;){
    bool addedTasks = false;
    
    if( quit_ ) {
      MPI_Cancel(&wakeUpRequest);
      cerrLock.lock(); mixedDebug << "Receiver " << proc_group_
				  << "-" << d_id << " quitting\n";
      cerrLock.unlock();
      TAU_PROFILE_EXIT( "thread quitting" );
      return;
    }

    ASSERT(recvs_.numRequests() > 0); // should have wake up request at least
    while (recvs_.waitsome(pg_, &semiReadyTasks_)) {
    }

    bool hasWakeUpSignal = false;
    list<int>::iterator iter = semiReadyTasks_.begin();
    while (iter != semiReadyTasks_.end()) {
      if (*iter == -1) {
	hasWakeUpSignal = true;
	iter = semiReadyTasks_.erase(iter);
	continue;
      }
      ++iter;
    }

    // Important to handle this wake up signal before handling the real
    // semiReadyTasks so that new tasks get added before checking to see
    // if their external recvs are finished.
    if (hasWakeUpSignal) {
      // clear out all wake-up signals since it's already awake
      int flag;	
      do {
	MPI_Iprobe(pg_->myrank(), d_id, pg_->getComm(), &flag, &status);
	if (flag) {
	  MPI_Recv(0, 0, MPI_INT, pg_->myrank(), d_id, pg_->getComm(),
		   &status);
	}
      } while (flag);
      
      // wake up signal -- may mean more tasks to add
      // (or can indicate externally requested receives came in)
      addedTasks = addAwaitingTasks();
      
      // post wake up signal receive again
      MPI_Irecv(0, 0, MPI_INT, pg_->myrank(), d_id, pg_->getComm(),
		&wakeUpRequest);
      recvs_.add(wakeUpRequest, 0, 0,"",0,-1);	
    }
  
    iter = semiReadyTasks_.begin();
    while (iter != semiReadyTasks_.end()) {
      int taskIndex = *iter;      
      ASSERT(taskIndex != -1); // these should have been removed
      if (awaitingTasks_[taskIndex]->isReady()) {
	// task has received all it needs and is ready to go
	d_parent->launchTask(awaitingTasks_[taskIndex]->getTask());
	delete awaitingTasks_[taskIndex];
	awaitingTasks_[taskIndex] = 0;
	availTaskSlots_.push(taskIndex);
	iter = semiReadyTasks_.erase(iter);
	continue;
      }
      ++iter;
    }

    if (recvs_.getUnfinishedBytes() != preBytesLeft || addedTasks) {
      // Number of bytes waiting for has changed -- reprioritize.
      if (d_parent->reprioritize(this, priorityItem_, addedTasks,
				 preBytesLeft)) {
	preBytesLeft = recvs_.getUnfinishedBytes();
      }
    }

    // check to see if there is anything to wait for
    d_lock.lock();
    if (availTaskSlots_.size() == awaitingTasks_.size() &&
	newTasks_.empty()) {
      pg_ = 0;
     d_lock.unlock();
      my_thread_->stop(); // wait till a task is added
     d_lock.lock();
      ASSERT(pg_ != 0);
      ASSERT(!newTasks_.empty());
    }
    d_lock.unlock();
  }
}

Receiver::AwaitingTask::
AwaitingTask(DetailedTask* task, list<DependencyBatch*> outstandingExtRecvs,
	     int threadID)
  : task_(task), outstandingExtRecvs_(outstandingExtRecvs)
{
  for (list<DependencyBatch*>::iterator iter = outstandingExtRecvs_.begin();
       iter != outstandingExtRecvs_.end(); ++iter) {
    (*iter)->addReceiveListener(threadID);
  }
}

bool Receiver::AwaitingTask::isReady() {
  list<DependencyBatch*>::iterator iter = outstandingExtRecvs_.begin();
  while (iter != outstandingExtRecvs_.end()) {
    if ((*iter)->wasReceived()) {
      iter = outstandingExtRecvs_.erase(iter);
    }
    else {
      ++iter;
    }
  }
  return outstandingExtRecvs_.empty();
}


/////////////// MPIReducer //////////////////

MPIReducer::MPIReducer(ThreadPool * parent)
  : d_parent( parent ), lock_("MPIReducer lock"), paused_(true), quit_(false)
{
  proc_group_ = Parallel::getRootProcessorGroup()->myrank();
}

void MPIReducer::assignTask( DetailedTask* task )
{
 lock_.lock();
  tasks_.push(task);
  if (paused_) {
    paused_ = false;
    my_thread_->resume();
  }
 lock_.unlock();
}

void MPIReducer::quit(  )
{
 lock_.lock();
  quit_ = true;
  if (paused_)
    my_thread_->resume();
 lock_.unlock();
}

void MPIReducer::run()
{
  TAU_REGISTER_THREAD();
  TAU_PROFILE("Reducer_run()", "void ()", TAU_DEFAULT);
  TAU_PROFILE_TIMER(doittimer, "doit Task", "[Reducer::run()]", TAU_DEFAULT);

  //if( mixedDebug.active() ) {
    cerrLock.lock();
    cerr /*mixedDebug*/ << "Reducer " << proc_group_
	       << ", Begin run() -- PID is " << getpid() << "\n";
    cerrLock.unlock();
    //}

  for (;;) {
   lock_.lock();
    if (tasks_.empty()) {
      paused_ = true;
     lock_.unlock();
      my_thread_->stop();
     lock_.lock();
    }
    if (quit_) {
      cerrLock.lock(); mixedDebug << "MPIReducer quitting\n";
      cerrLock.unlock();
      TAU_PROFILE_EXIT( "thread quitting" );
      return;  
    }
    ASSERT(!tasks_.empty());
    DetailedTask* reductionTask = tasks_.front();
    tasks_.pop();
   lock_.unlock();
   ASSERT(reductionTask != 0);
   d_parent->getScheduler()->runReductionTask(reductionTask);
   d_parent->doneReduce(reductionTask);
  }
}

/////////////// ThreadPool //////////////////

ThreadPool::ThreadPool( MixedScheduler* scheduler,
			int maxReceivers, int maxWorkers ) :
  d_scheduler( scheduler ),
  d_maxWorkers( maxWorkers ),
  d_maxReceivers( maxReceivers ),
  d_numBusy( 0 ),
  d_numWaiting( 0 ),
  d_numReduces( 0 ),
  d_numReduced( 0 ),
  d_receiverQueueLock("ThreadPool ReceiverQueue lock")
{
  d_waitingForReprioritizingThread = d_waitingTilDoneThread = 0;
  d_numReceiversAddingTasks = d_maxBytesWhileAddingTasks = 0;
  d_beginTime = Time::currentSeconds();

  d_timeUsed = scinew double[ maxWorkers ];

  //if( mixedDebug.active() ) {
    cerrLock.lock();
    cerr/*mixedDebug*/ << "Main " << Parallel::getRootProcessorGroup()->myrank()
	       << " -- PID is " << getpid() << "\n";
    cerrLock.unlock();
    //}
  
  char name[1024];
  
  num_workers_ = scinew Semaphore( "number of threads", maxWorkers );

  // Only one thing (worker thread or threadPool itself) can be
  // modifying the workerQueue (actually a stack) at a time.
  d_workerQueueLock = scinew Mutex( "ThreadPool Worker Queue Lock" );

  d_workerReadyLocks = scinew Mutex*[ maxWorkers ];

  d_workers = scinew Worker*[ maxWorkers ];

  for( int i = 0; i < maxWorkers; i++ ){

    d_timeUsed[ i ] = 0.0;

    d_availableWorkers.push( i );

    d_workerReadyLocks[ i ] = scinew Mutex( "Worker Ready Lock" );
    // None of the workers are allowed to run until the ThreadPool
    // tells them to run... therefore they must be locked initially.
    d_workerReadyLocks[ i ]->lock();

    Worker * worker = scinew Worker( this, i,
				     d_workerReadyLocks[ i ] );

    sprintf( name, "Worker %d-%d",
	     Parallel::getRootProcessorGroup()->myrank(), i );

    Thread * t = scinew Thread( worker, name );
    t->detach();

    d_workers[ i ] = worker;
  }

  ReceiverPriorityQueue::iterator initReceivers =
    d_receiverQueue.insert(make_pair(0 /* zero bytes receiving so far */,
				     scinew list<Receiver*>())).first;
  list<Receiver*>* initReceiversList = initReceivers->second;
  d_receivers = scinew Receiver*[ maxReceivers ];
  for (int i = 0; i < maxReceivers; ++i) {
    d_receivers[i] = scinew Receiver(this, i);
    list<Receiver*>::iterator posIter = 
      initReceiversList->insert(initReceiversList->end(), d_receivers[i]);
    ReceiverPriorityQueueItem priorityItem =
      make_pair(initReceiversList, posIter);
    
    // tell the receiver where it is in the priority queue
    d_receivers[i]->setPriorityItem(priorityItem);
    
    sprintf( name, "Receiver %d-%d",
	     Parallel::getRootProcessorGroup()->myrank(), i );

    Thread * t = scinew Thread(d_receivers[i], name);
    t->detach();
  }
  sprintf( name, "MPIReducer %d",
	   Parallel::getRootProcessorGroup()->myrank());
  d_reducer = scinew MPIReducer(this);
  Thread * t = scinew Thread(d_reducer, name);
  t->detach();
  
  ASSERT(d_waitingForReprioritizingThread == 0);
  Thread::yield(); // let other threads start ??? 
}

ThreadPool::~ThreadPool()
{
  // Wait for all threads to be finished doing what they were doing.
  cout << "in destructor: Waiting for all threads to finish\n";
  while( d_numWaiting + d_numBusy > 0 )
    {
      Time::waitFor(.001);
    }

  cout << "Telling all threads to quit\n";
  for( int i = 0; i < d_maxReceivers; i++ ){
    d_receivers[ i ]->quit();
  }  
  for( int i = 0; i < d_maxWorkers; i++ ){
    d_workers[ i ]->quit();
  }
}

double
ThreadPool::getUtilization()
{
  double time = Time::currentSeconds();
  double totalTimeUsed = 0;
  double maxTime = d_maxWorkers * ( time - d_beginTime );

  d_workerQueueLock->lock();

  for( int i = 0; i < d_maxWorkers; i++ ){
    totalTimeUsed += d_timeUsed[ i ];
  }

  d_workerQueueLock->unlock();

  return totalTimeUsed / maxTime;
}

int
ThreadPool::available()
{
  d_workerQueueLock->lock();
  int numAvail = d_maxWorkers - d_numBusy;

#if DAV_DEBUG
  static int cnt = 0;
  static double avg = 0;
  static int minFree = 99999;
  if( numAvail < minFree ){
    minFree = numAvail;
  }
  cnt++;
  avg += d_numBusy;
  if( cnt % 50000 == 0 ){
    cerr << "ThreadPool: number of threads available: " 
	 << d_maxWorkers - d_numBusy << ", min: " << minFree << "\n";
    cerr << "            avg busy: " << avg / cnt << "\n";
  }
#endif

  d_workerQueueLock->unlock();

  return numAvail;
}

void
ThreadPool::addTask( DetailedTask* task )
{
  // use addReductionTask for reduction tasks
  ASSERT(task->getTask()->getType() != Task::Reduction);  
 d_receiverQueueLock.lock();
  d_numWaiting++; 
  ReceiverPriorityQueue::iterator head_iter;
  for (head_iter = d_receiverQueue.begin();
       head_iter != d_receiverQueue.end(); ++head_iter) {
    if (!head_iter->second->empty() &&
	head_iter->second->front() != 0) break;
  }
  if (head_iter == d_receiverQueue.end() ||
      (d_numReceiversAddingTasks > 0 &&
       head_iter->first > d_maxBytesWhileAddingTasks)) {
    // need to wait for receivers to be reprioritized
    d_waitingForReprioritizingThread = Thread::self();
   d_receiverQueueLock.unlock();
    d_waitingForReprioritizingThread->stop();
   d_receiverQueueLock.lock(); 

    for (head_iter = d_receiverQueue.begin();
	 head_iter != d_receiverQueue.end(); ++head_iter) {
      if (!head_iter->second->empty() &&
	  head_iter->second->front() != 0) break;
    }
    ASSERT(head_iter != d_receiverQueue.end()); // should have one ready 
    ASSERT(d_numReceiversAddingTasks == 0 ||
	   head_iter->first <= d_maxBytesWhileAddingTasks);       
  }

  list<Receiver*>* topPriority = head_iter->second;
  ASSERT(!topPriority->empty());
  Receiver* chosenReceiver = *topPriority->begin();
  chosenReceiver->assignTask(task);

  // take it out of the list -- don't assign to it until reprioritized
  topPriority->pop_front();
  // put a null place holder at the end so it doesn't delete the list
  // until reprioritized
  topPriority->push_back(0);

  ++d_numReceiversAddingTasks;

  // only receivers with unfinishedBytes <= d_maxBytesWhileAddingTasks may be
  // used until the ones taken off for adding tasks have been reprioritized.
  d_maxBytesWhileAddingTasks = head_iter->first;

 d_receiverQueueLock.unlock();
}

void
ThreadPool::addReductionTask( DetailedTask* task )
{
  ASSERT(task->getTask()->getType() == Task::Reduction);
  ++d_numReduces;
  d_reducer->assignTask(task);
}

bool
ThreadPool::reprioritize( Receiver* receiver, ReceiverPriorityQueueItem& item,
			  bool tasksAdded, int oldBytes )
{
  static set<int> emptyBins;

 d_receiverQueueLock.lock();  
  if (receiver->hasNewTasks()) {
    // wait til the tasks are added before adding it back into the queue
    d_receiverQueueLock.unlock();    
    return false;
  }
  
  list<Receiver*>* oldList = item.first;
  int bytes = receiver->getUnfinishedBytes();

  // remove the item from its previous location
  if (tasksAdded) {
    // remove null place holder at the end (doesn't matter which one if there
    // are several, as long as they are all at the end).
    ASSERTEQ(oldList->back(), 0);
    oldList->pop_back();
  }
  else {
    oldList->erase(item.second);
  }

  if (oldBytes == receiver->getUnfinishedBytes()) {
    // should only reprioritize if the bytes has changed or tasks were added
    ASSERT(tasksAdded);

    // just add it back to its original list
    oldList->push_front(receiver);  // go in front, keep nulls in back
    item.second = oldList->begin();
  }
  else {
    // see if the old list is now empty and should be removed
    if (oldList->empty()) {
      ReceiverPriorityQueue::iterator toRemove =
	d_receiverQueue.find(oldBytes);
      ASSERT(toRemove != d_receiverQueue.end());
      ASSERTEQ(toRemove->second, oldList);
      delete oldList;
      d_receiverQueue.erase(toRemove);
    }  
    
    // insert the item into a new location
    list<Receiver*>* receiverList = scinew list<Receiver*>();
    pair<ReceiverPriorityQueue::iterator, bool> insertResult = 
      d_receiverQueue.insert(make_pair(bytes, receiverList));
    if (!insertResult.second) {
      delete receiverList; // list already existed
      receiverList = insertResult.first->second;
    }
    receiverList->push_front(receiver); // go in front, keep nulls in back
    // set new location
    item = make_pair(receiverList, receiverList->begin());
  }
  
  if (tasksAdded) {
    // reprioritized one of the recently assigned-to receivers
    --d_numReceiversAddingTasks;
  }
  
  if (d_waitingForReprioritizingThread != 0 &&
      (bytes < d_maxBytesWhileAddingTasks || d_numReceiversAddingTasks == 0)){
    d_waitingForReprioritizingThread->resume();
    d_waitingForReprioritizingThread = 0;
  }

#if 0    
  /* for testing */
  cerrLock.lock();
  for (ReceiverPriorityQueue::iterator iter = d_receiverQueue.begin();
       iter != d_receiverQueue.end(); ++iter) {
    for (list<Receiver*>::iterator recvListIter = iter->second->begin();
	 recvListIter != iter->second->end(); ++recvListIter) {
      if (*recvListIter != 0) {
	cerr << iter->first << " ";
      }
    }
  }
  cerr << "\n";
  cerrLock.unlock();
  /* end for testing */
#endif
  
 d_receiverQueueLock.unlock();
 return true;
}

void
ThreadPool::launchTask( DetailedTask          * task )
{
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "ThreadPool pre-assign worker\n";
    cerrLock.unlock();
  }
#endif

  num_workers_->down();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug<< "ThreadPool: worker available\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->lock();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "ThreadPool now-assign worker\n";
    cerrLock.unlock();
  }
#endif

  //  if( d_availableWorkers.empty() ){
  //    throw( InternalError( "All threads busy..." ) );
  //  }
  
  Worker * worker;
  int      id = d_availableWorkers.top();

  d_availableWorkers.pop();

  worker = d_workers[ id ];

  d_numBusy++;
 d_receiverQueueLock.lock();  
  d_numWaiting--;
 d_receiverQueueLock.unlock();
  worker->assignTask( task );

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Unlocking Worker Thread " << id << "\n";
    cerrLock.unlock();
  }
#endif

  d_workerReadyLocks[ id ]->unlock(); // start worker

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker Thread " << id << " unlocked\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->unlock();
}

void
ThreadPool::all_done()
{
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "waiting for all threads to finish\n";
    cerrLock.unlock();
  }

  if (d_numBusy + d_numWaiting > 0 || d_numReduces > d_numReduced) {
    d_waitingTilDoneThread = Thread::self();      
    d_waitingTilDoneThread->stop();
  }
  ASSERT(d_numBusy + d_numWaiting == 0);
  ASSERT(d_numReduced == d_numReduces);

  // if( mixedDebug.active() ) {
  cerrLock.lock();
  cerr << "all threads now done: d_numBusy is " << d_numBusy
       << " d_numWaiting is " << d_numWaiting << "\n";
  cerrLock.unlock();
  //}
}

void
ThreadPool::done( int id, DetailedTask * task, double timeUsed )
{
  
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug<< "Worker " << id << " done.  getting queue lock\n";
    cerrLock.unlock();
  }
#endif

  d_workerQueueLock->lock();

#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << id << " finished: " << *task << "\n";
    cerrLock.unlock();
  }
#endif

  d_timeUsed[ id ] += timeUsed;

  d_availableWorkers.push( id );
  --d_numBusy;

  if (d_waitingTilDoneThread != 0) {
    if (d_numReduced == d_numReduces && d_numBusy + d_numWaiting == 0) {
      d_waitingTilDoneThread->resume();
      d_waitingTilDoneThread = 0;
    }
  }

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "numBusy: " << d_numBusy << "\n"; 
    cerrLock.unlock();
  }

  num_workers_->up();

  d_workerQueueLock->unlock();
#if DAV_DEBUG
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Worker " << id 
	       << " added back to the list of threads available\n";
    cerrLock.unlock();
  }
#endif
}

void ThreadPool::doneReduce(DetailedTask * /*task*/)
{
  ++d_numReduced;
  if (d_waitingTilDoneThread != 0) {
    if (d_numReduced == d_numReduces && d_numBusy + d_numWaiting == 0) {
     d_workerQueueLock->lock(); // prevent from double resuming
      if (d_numBusy + d_numWaiting == 0) {
	d_waitingTilDoneThread->resume();
	d_waitingTilDoneThread = 0;
      }
     d_workerQueueLock->unlock();
    }
  }
}


