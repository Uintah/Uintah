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

#if !defined(__APPLE__) && !defined(__sgi)
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
// We mustn't call the PMPI routines when using TAU, or they won't get profiled.
#if defined(LAM_MPI) && !defined(USE_TAU_PROFILING)
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Malloc/Allocator.h>
#include <sys/time.h>
#include <Core/Parallel/MPI_Communicator.h>

#define debug_main
#define debug_main_thread
#define debug_mpi_thread


using std::cout;
using std::endl;
using namespace SCIRun;

#ifdef MALLOC_TRACE
#include "MallocTraceMPIOff.h"
#endif


MpiCall::MpiCall(){
}

MpiCall::~MpiCall(){
}

//-------------------------- MPI_Init_thread ------------------------
// we use the PMPI_*() functions in libmpi
extern "C" int PMPI_Init_thread(int *, char***, int, int*);

// we export MPI_*() functions, intercept and wrap them using the PMPI_*() functions
extern "C" {
  extern int MPI_Init_thread(int *, char***, int, int*);
}

Init_threadMpiCall::Init_threadMpiCall(int* argc, char*** argv,
				       int threadlevel, int* thread_supported, int* retval)
  : argc_(argc), argv_(argv),
    threadlevel_(threadlevel), thread_supported_(thread_supported), retval_(retval)
{
}

Init_threadMpiCall::~Init_threadMpiCall()
{
}

int MPI_Init_thread(int* argc, char*** argv,
		    int threadlevel, int* thread_supported){

  int threadmpi = 0;
  int argc_in = *argc;
  char** argv_in = *argv;
  for(int i=1;i<argc_in;i++){
    string s=argv_in[i];
    if( (s == "-threadmpi") ) {
      threadmpi = 1;
    }
  }
  if ( threadmpi ){
    int retval = 0;
    cout<<"Using seperate MPI thread for async communication"<<endl;
    Init_threadMpiCall *myCall = scinew Init_threadMpiCall(argc,argv,threadlevel,thread_supported, &retval);
    MPICommObj.schedule(myCall);
    delete myCall;
    return retval;
  }
  else{
    return PMPI_Init_thread(argc,argv,threadlevel,thread_supported);
  }
}

//-------------------------- MPI_Finalize ------------------------
// we use the PMPI_*() functions in libmpi
extern "C" int PMPI_Finalize();

// we export MPI_*() functions, intercept and wrap them using the PMPI_*() functions
extern "C" {
  extern int MPI_Finalize();
}

FinalizeMpiCall::FinalizeMpiCall(int* retval)
  : retval_(retval)
{
}

FinalizeMpiCall::~FinalizeMpiCall()
{
}

int MPI_Finalize(){
  if (pthread_getspecific(MPICommObj.thread_running)){
    int retval = 0;
    //cout<<"Call to MPI_Finalize intercepted by library"<<endl;
    FinalizeMpiCall *myCall = scinew FinalizeMpiCall(&retval);
    MPICommObj.schedule(myCall);
    delete myCall;
    return retval;
  }
  else{
    return PMPI_Finalize();
  }
}
//-------------------------------------------------------------

int MPI_Communicator::schedule(MpiCall* someCall){
  int retval = 0;
  if (someCall){
    this->mutey.lock();{
      this->mpiCallQueue.push(someCall);
      this->MPICallQueue.conditionSignal();
      this->blockingMPICall.wait(this->mutey);
    }
    this->mutey.unlock();
  }
  else{
    cout<<"Null pointer passed to MPI_Communicator::schedule()"<<endl;
    exit(1);
  }
  return retval;
}

int MPI_Communicator::schedule(Init_threadMpiCall* someCall){
  int retval = 0;
  if (someCall){
    this->mutey.lock();{
      this->mpiCallQueue.push(someCall); //the MPI thread does not do the call
      someCall->mainexecute();           //we make the call ourselves here
      MPIThread = scinew Thread(this, "Handle MPI calls asynchronously");
      pthread_setspecific(this->thread_running, (const void*)1);
      this->MPICallQueue.conditionSignal();
      this->blockingMPICall.wait(this->mutey);
    }
    this->mutey.unlock();
  }
  else{
    cout<<"Null pointer passed to MPI_Communicator::schedule()"<<endl;
    exit(1);
  }
  return retval;
}

int MPI_Communicator::schedule(FinalizeMpiCall* someCall){
  int retval = 0;
  if (someCall){
    this->mutey.lock();{
      this->mpiCallQueue.push(someCall);
      this->MPICallQueue.conditionSignal();
      this->blockingMPICall.wait(this->mutey);
    }
    this->mutey.unlock();
    
    this->MPIThread->join();
    pthread_setspecific(this->thread_running,(const void*)0);
    someCall->mainexecute();
    //delete MPIThread; destructor is private, so this does not work...
  }
  else{
    cout<<"Null pointer passed to MPI_Communicator::schedule()"<<endl;
    exit(1);
  }
  return retval;
}

void MPI_Communicator::run()
{
  int kill_mpi_thread = 0;
  mutey.lock();
  //our first call had better be some sort of init call
  this->mpiCallQueue.front()->execute();
  this->mpiCallQueue.pop();
  this->blockingMPICall.conditionSignal();
  this->mutey.unlock();
  
  struct timespec at;
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  at.tv_sec  = tv.tv_sec;
  at.tv_nsec = tv.tv_usec * 1000;
  at.tv_sec += 1;
  
  while(!kill_mpi_thread){
    mutey.lock();
    if ( this->mpiCallQueue.empty() ){
      //trigger progress for outstanding Isend calls
      int probe_flag = 0;
      MPI_Status Pstatus[1];
      for (int j = 0; j < this->num_iprobes; j++){
	PMPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&probe_flag,Pstatus);
      }
    }
    else{
      this->mpiCallQueue.front()->execute();
      if (dynamic_cast<FinalizeMpiCall*>(this->mpiCallQueue.front())){
      	kill_mpi_thread = 1;
      }
      this->mpiCallQueue.pop();
      this->blockingMPICall.conditionSignal();

      if (kill_mpi_thread){
	mutey.unlock();
      	Thread::exit();
      }
    }
    
    gettimeofday(&tv, &tz);
    at.tv_sec  = tv.tv_sec;
    at.tv_nsec = tv.tv_usec * 1000;
    at.tv_nsec += this->wait_time * 1000;
    
    //implicit mutey->unlock()
    if (MPICallQueue.timedWait(mutey,&at) == false ){
      //implicit mutey->lock()
      mutey.unlock();
    }
    else{
      //implicit mutey->lock()
      mutey.unlock();
    }
  }
  Thread::exit();
}

MPI_Communicator::MPI_Communicator()
  : Runnable(false), //don't delete an MPI_Communicator object on thread exit
  mutey("MPI call queue"),
  blockingMPICall("wait in main thread on a blocking MPI call"),
  MPICallQueue("queue used to pass work from main thread to MPI thread"),
  wait_time(WAIT_TIME_DEFAULT),
  num_iprobes(NUM_IPROBES_DEFAULT)
{
  if(pthread_key_create(&this->thread_running, NULL) != 0){
    cout<<"could not create key in MPI_Communicator"<<"\n";
  }
  pthread_setspecific(this->thread_running,(const void*)0);
}

MPI_Communicator::~MPI_Communicator(){
}

#define MPI_CLASS_PRE(N) N ## MpiCall
#define MPI_CALL_PRE(N) MPI_ ## N
#define PMPI_CALL_PRE(N) PMPI_ ## N
#define MPI_CLASS(N) MPI_CLASS_PRE(N)
#define MPI_CALL(N) MPI_CALL_PRE(N)
#define PMPI_CALL(N) PMPI_CALL_PRE(N)

#define MPI_CLASS_BODY \
extern "C" RET_TYPE PMPI_CALL(NAME)(CALLSIG); \
extern "C" { \
  extern RET_TYPE MPI_CALL(NAME)(CALLSIG); \
} \
\
class MPI_CLASS(NAME) : public MpiCall { \
public: \
  virtual int execute(){ \
    *(this->retval) =  PMPI_CALL(NAME)(CALLARGS); \
    return 0; \
  }\
\
  MPI_CLASS(NAME)(CALLSIG, RET_TYPE *retval);\
  ~MPI_CLASS(NAME)();\
\
private:\
  VARS\
  RET_TYPE *retval; \
};\
\
MPI_CLASS(NAME)::MPI_CLASS(NAME)(CALLSIG, RET_TYPE *retval)\
  : VAR_INIT \
{\
}\
\
MPI_CLASS(NAME)::~MPI_CLASS(NAME)(){\
}\
\
RET_TYPE MPI_CALL(NAME)(CALLSIG){\
  if (pthread_getspecific(MPICommObj.thread_running)){\
    RET_TYPE retval;\
    MPI_CLASS(NAME) *myCall = scinew MPI_CLASS(NAME)(CALLARGS, &retval);\
    MPICommObj.schedule(myCall);\
    delete myCall;\
    return retval;\
  }\
  else{\
    return PMPI_CALL(NAME)(CALLARGS);\
  }\
} \

#include <Core/Parallel/stubs.h>

#ifdef MALLOC_TRACE
#include "MallocTraceMPIOn.h"
#endif

#endif
#endif
