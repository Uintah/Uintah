#include <Core/Thread/Thread.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/ThreadGroup.h>
//#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
using namespace SCIRun;

class ParallelWorker {
  WorkQueue work;
  int * buffer;
  int buff_size;
  int np;
public:
  ParallelWorker(): work(0), buffer(0), buff_size(0) {}
  ParallelWorker(int buff_size, int np, int granularity):
    work("ParallelWorker::WorkQueue"),
    buffer(0), buff_size(buff_size), np(np)
  {
    buffer = (int *)malloc(sizeof(int)*buff_size);
    for(int i = 0; i < buff_size; i++)
      buffer[i] = -1;
    work.refill(buff_size,np,granularity);
  }
  ~ParallelWorker() {
    if (buffer)
      free(buffer);
  }
  
  void do_work(int proc) {
    //cerr << "Proc: "<<proc<<" of "<<np<<" working\n";
    cerr << proc;
#if 0
    int start = (int)((float)buff_size / np * proc);
    int end = (int)((float)buff_size / np * (proc+1));
    if (end > buff_size) end = buff_size;
    for (int i = start; i < end; i++)
      buffer[i] = proc;
#else
    int start, end;
    while(work.nextAssignment(start,end)) {
      for (;start < end; start++)
	buffer[start] = proc;
    }
#endif
  }

  void print_test() {
    if (buff_size <= 0) {
      cout << "ParallelWorker is empty\n";
      return;
    }
    int prev = buffer[0];
    cout << "ParallelWorker: buff_size("<<buff_size<<"), np("<<np<<")\n";
    cout << "buffer[0] = "<<prev<<endl;
    for (int i = 0; i < buff_size; i++) {
      if (prev != buffer[i]) {
	prev = buffer[i];
	cout << "buffer["<<i<<"] = "<<prev<<endl;
      }
    }
  }
};

int main(int argc, char *argv[]) {
  int np = 1;
  int buff_size = 100;
  int granularity = 5;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i],"-np") == 0) {
      i++;
      np = atoi(argv[i]);
    } else if (strcmp(argv[i],"-size") == 0) {
      i++;
      buff_size = atoi(argv[i]);
    } else if (strcmp(argv[i],"-granularity") == 0) {
      i++;
      granularity = atoi(argv[i]);
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
  if (granularity < 0) granularity = 5;
  cout <<"np = "<<np<<", buff_size = "<<buff_size<<", granularity = "<<granularity<<endl;
  
  ParallelWorker worker(buff_size,np,granularity);
  Parallel<ParallelWorker> phelper(&worker, &ParallelWorker::do_work);
  Thread::parallel(phelper, np, true);

  worker.print_test();
  cerr << "Program end\n";
  Thread::exitAll(0);

  return 0;
}
