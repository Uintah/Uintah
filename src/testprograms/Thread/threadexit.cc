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

#include <Core/Thread/Thread.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/ThreadGroup.h>
//#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

class Globals {
public:
  Globals(): group(0), work(0), mutex(0), np(0), exit_everybody(false) {}
  Globals(SCIRun::ThreadGroup *group, SCIRun::WorkQueue *work, SCIRun::Mutex *mutex, SCIRun::Barrier *barrier,
	  const int np):
    group(group), work(work), mutex(mutex), barrier(barrier), np(np),
    exit_everybody(false) {}
  SCIRun::ThreadGroup *group;
  SCIRun::WorkQueue *work;
  SCIRun::Mutex *mutex;
  SCIRun::Barrier *barrier;
  int np;

  // exit stuff
  bool exit_everybody;
  int exit_status;
  void exit_clean(int status = 0) {
    exit_everybody = true;
    exit_status = status;
  }
  bool stop_execution() { return exit_everybody; }

};

class Display;

class Worker: public SCIRun::Runnable {
public:
  Worker(const int procID, Display *dpy, Globals *globals):
    procID(procID), dpy(dpy), globals(globals)
  {}
  int procID;
  Display *dpy;
  Globals *globals;

  void run();
};

class Display: public SCIRun::Runnable {
public:
  Display(const int buff_size, Globals *globals, const int stopper) :
    buff_size(buff_size), 
    globals(globals), 
    stopper(stopper)
  {
    buffer = new int[buff_size];
    buffer_display = new int[buff_size];
    for(int i = 0; i < buff_size; i++)
      buffer[i] = -1;
  }
  ~Display() {
    if (buffer)
      delete[] buffer;
    if (buffer_display)
      delete[] buffer_display;
  }

  int buff_size;
  Globals *globals;
  int stopper;
  int *buffer;
  int *buffer_display;


  void print_test() {
    if (buff_size <= 0) {
      std::cout << "ParallelWorker is empty\n";
      return;
    }
    int prev = buffer_display[0];
    std::cout << "ParallelWorker: buff_size("<<buff_size<<"), np("<<globals->np<<")\n";
    std::cout << "buffer_display[0] = "<<prev<<std::endl;
    for (int i = 0; i < buff_size; i++) {
      if (prev != buffer_display[i]) {
	prev = buffer_display[i];
	std::cout << "buffer_display["<<i<<"] = "<<prev<<std::endl;
      }
    }
  }

  void run() {

    for(int i = 0;;i++) {
      // sync all the workers
      globals->barrier->wait(globals->np + 1);
      if (globals->stop_execution()){
        //        for(;;) {}
        //        SCIRun::Thread::exit();
        return;
      }

      // refill the workqueue
      globals->work->refill(buff_size, globals->np);
      // copy the data
      memcpy(buffer_display, buffer, buff_size * sizeof(int));
      
      // sync all the workers again who should be ready to work
      globals->barrier->wait(globals->np + 1);

      // let the workers work while we display
      print_test();

      if (i == stopper) {
	//	SCIRun::Thread::exitAll(0);
	globals->exit_clean(0);
      }
    }
  }
};
  
void Worker::run() {
  for (;;) {
    // sync all the workers and display
    globals->barrier->wait(globals->np + 1);
    if (globals->stop_execution()){
      //      for(;;) {}
      //      SCIRun::Thread::exit();
      return;
    }
    // let the display refill the work queue
    globals->barrier->wait(globals->np + 1);

    // do the work
    int start = 0;
    int end = -1;
    while(globals->work->nextAssignment(start,end)) {
      for(int i = start; i < end; i++)
	dpy->buffer[i] = procID;
    }
  }
}

int main(int argc, char *argv[]) {
  int np = 1;
  int buff_size = 100;
  int stopper = 10;
  
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i],"-np") == 0) {
      i++;
      np = atoi(argv[i]);
    } else if (strcmp(argv[i],"-size") == 0) {
      i++;
      buff_size = atoi(argv[i]);
    } else if (strcmp(argv[i],"-stop") == 0) {
      i++;
      stopper = atoi(argv[i]);
    } else {
      std::cout << "parallel -np [int] -size [int]\n";
      std::cout << "-np\tnumber of processors/helpers to use.\n";
      std::cout << "-size\tsize of array to use\n";
      return 1;
    }
  }

  // validate parameters
  if (np < 1) np = 1;
  if (buff_size < 1) buff_size = 100;
  std::cout <<"np = "<<np<<", buff_size = "<<buff_size<<std::endl;

  SCIRun::ThreadGroup *group = new SCIRun::ThreadGroup("threadexit group");
  SCIRun::WorkQueue *work = new SCIRun::WorkQueue("threadexit workqueue");
  SCIRun::Mutex *mutex = new SCIRun::Mutex("threadexit mutex");
  SCIRun::Barrier *barrier = new SCIRun::Barrier("threadexit barrier");
  Globals *globals = new Globals(group, work, mutex, barrier, np);
  Display *dpy = new Display(buff_size, globals, stopper);
  new SCIRun::Thread(dpy, "Display thread", group);
  for(int i = 0; i < np; i++) {
    char buf[100];
    sprintf(buf, "worker %d", i);
    new SCIRun::Thread(new Worker(i, dpy, globals), buf, group);
  }
#if 0
  group->detach();
#else
   group->join();
   std::cout << "Threads exited" << std::endl;
#endif
   
  return 0;
}
