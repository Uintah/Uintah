
#define USE_SCI_THREADS

#ifdef USE_SCI_THREADS
#include <Core/Thread/Thread.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Runnable.h>
#else
#include "Thread/Thread.h"
#include "Thread/Barrier.h"
#include "Thread/Runnable.h"
#endif

#include <string.h>
#include <iostream>

extern void run_gl_test();

#ifdef USE_SCI_THREADS
using namespace SCIRun;
#else
#endif
using namespace std;


class GL_Test : public Runnable {
public:
  virtual ~GL_Test() {}

  virtual void run() {
    run_gl_test();
  }
};

class Spin : public Runnable {
public:
  virtual ~Spin() {}

  virtual void run() {
    for (;;) {}
  }
};

#define MAX_STR_LEN 128

class SpinPrint : public Runnable {
  char my_str[MAX_STR_LEN];
public:
  SpinPrint(char *str) {
    strncpy(my_str,str,MAX_STR_LEN-1);
  }
  virtual ~SpinPrint() {}

  virtual void run() {
    for (int i=0;;i++) {
      if (i%10000 == 0)
	cout << my_str;
    }
  }
};

class Wait : public Runnable {
  int procs;
  Barrier *barrier;
  long spin_count;
public:
  Wait(int _procs, Barrier *_barrier, long _spin_count=0):
    procs(_procs), barrier(_barrier), spin_count(_spin_count) {}
  virtual ~Wait() {}

  virtual void run() {
    for (;;) {
      for (long i = 0; i < spin_count; i++);
#ifdef USE_SCI_THREADS
      barrier->wait(procs);
#else
      barrier->wait();
#endif
    }
  }
};

int main() {
  int num_waiters = 3;
#ifdef USE_SCI_THREADS
  Barrier *barrier = new Barrier("glthread");
#else
  Barrier *barrier = new Barrier("glthread",num_waiters);
#endif

  //  run_gl_test();

  new Thread(new GL_Test(), "GL_Test_1");
  //new Thread(new GL_Test(), "GL_Test_2");
  new Thread(new Spin(), "Spin_1");
  new Thread(new Spin(), "Spin_2");
  //new Thread(new SpinPrint("+"), "SpinPrint_1");

  new Thread(new Wait(num_waiters, barrier), "Wait_1");
  new Thread(new Wait(num_waiters, barrier, 1e9), "Wait_2");
  //  new Thread(new Wait(num_waiters, barrier, 5e9), "Wait_3");
  
  return 0;
}
