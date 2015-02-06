/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace std;
using namespace SCIRun;

class Globals {
public:
  Globals(): group(0), work(0), mutex(0), np(0), exit_everybody(false) {}
  Globals(ThreadGroup *group, WorkQueue *work, Mutex *mutex, Barrier *barrier,
	  const int np):
    group(group), work(work), mutex(mutex), barrier(barrier), np(np),
    exit_everybody(false) {}
  ThreadGroup *group;
  WorkQueue *work;
  Mutex *mutex;
  Barrier *barrier;
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

class Worker: public Runnable {
public:
  Worker(const int procID, Display *dpy, Globals *globals):
    procID(procID), dpy(dpy), globals(globals)
  {}
  int procID;
  Display *dpy;
  Globals *globals;

  void run();
};

class Display: public Runnable {
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
      cout << "ParallelWorker is empty\n";
      return;
    }
    int prev = buffer_display[0];
    cout << "ParallelWorker: buff_size("<<buff_size<<"), np("<<globals->np<<")\n";
    cout << "buffer_display[0] = "<<prev<<endl;
    for (int i = 0; i < buff_size; i++) {
      if (prev != buffer_display[i]) {
	prev = buffer_display[i];
	cout << "buffer_display["<<i<<"] = "<<prev<<endl;
      }
    }
  }

  void run() {

    for(int i = 0;;i++) {
      // sync all the workers
      globals->barrier->wait(globals->np + 1);
      if (globals->stop_execution()){
        //        for(;;) {}
        //        Thread::exit();
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
	//	Thread::exitAll(0);
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
      //      Thread::exit();
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
      cout << "parallel -np [int] -size [int]\n";
      cout << "-np\tnumber of processors/helpers to use.\n";
      cout << "-size\tsize of array to use\n";
      return 1;
    }
  }

  // validate parameters
  if (np < 1) np = 1;
  if (buff_size < 1) buff_size = 100;
  cout <<"np = "<<np<<", buff_size = "<<buff_size<<endl;

  ThreadGroup *group = new ThreadGroup("threadexit group");
  WorkQueue *work = new WorkQueue("threadexit workqueue");
  Mutex *mutex = new Mutex("threadexit mutex");
  Barrier *barrier = new Barrier("threadexit barrier");
  Globals *globals = new Globals(group, work, mutex, barrier, np);
  Display *dpy = new Display(buff_size, globals, stopper);
  new Thread(dpy, "Display thread", group);
  for(int i = 0; i < np; i++) {
    char buf[100];
    sprintf(buf, "worker %d", i);
    new Thread(new Worker(i, dpy, globals), buf, group);
  }
#if 0
  group->detach();
#else
   group->join();
   cout << "Threads exited" << endl;
#endif
   
  return 0;
}
