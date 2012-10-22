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

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <queue>
#include <unistd.h>
#include <cstdlib>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <sys/time.h>

int WAIT_TIME_DEFAULT   = 10000; //how many microseconds to sleep before triggering the progress loop
int NUM_IPROBES_DEFAULT = 15;    //how many Iprobes to do inside of the progress loop


using namespace SCIRun;

class MpiCall {
public:
  MpiCall();
  virtual ~MpiCall();
  virtual int execute() = 0;
};

class Init_threadMpiCall : public MpiCall {
public:
  virtual int execute(){
    //this is the code executed by the MPI thread
    return 0;
  }
  virtual int mainexecute(){
    //this is the code executed by our main thread
    *(this->retval_)  = PMPI_Init_thread(argc_,
			argv_,
			threadlevel_,
			thread_supported_);
    return 0;
  }
  
  Init_threadMpiCall(int* argc, char*** argv,
		     int threadlevel, int* thread_supported, int* retval);
  ~Init_threadMpiCall();
    
private:
  int* argc_;
  char*** argv_;
  int threadlevel_;
  int* thread_supported_;

  int* retval_;
};

class FinalizeMpiCall : public MpiCall {
public:
  virtual int execute(){
    //this is the code executed by the MPI thread
    return 0;
  }
  virtual int mainexecute(){
    //this is the code executed by our main thread
    *(this->retval_)  = PMPI_Finalize();
    return 0;
  }
  
  FinalizeMpiCall(int* retval);
  ~FinalizeMpiCall();
    
private:
  int* retval_;
};

class MPI_Communicator : public Runnable {
public:

  MPI_Communicator();
  ~MPI_Communicator();

  void run();
  int schedule(MpiCall* someCall);
  int schedule(Init_threadMpiCall* someCall);
  int schedule(FinalizeMpiCall* someCall);
  
  pthread_key_t   thread_running;

private:
  std::queue<MpiCall*> mpiCallQueue;
  Thread *MPIThread;

  Mutex mutey; //mutey("MPI call queue");
  ConditionVariable blockingMPICall; //used by main thread to block on a blocking MPI call
  ConditionVariable MPICallQueue;  //used by MPI thread to wakeup when there is an a new
                                   //MPI call in the Queue
  int wait_time;
  int num_iprobes;
};



MPI_Communicator MPICommObj;
