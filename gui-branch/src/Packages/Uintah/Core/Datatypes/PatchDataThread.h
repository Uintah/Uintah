#ifndef UINTAH_PATCHDATATHREAD_H
#define UINTAH_PATCHDATATHREAD_H

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>


#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;
namespace Uintah {

using SCIRun::Thread;
using SCIRun::ThreadGroup;
using SCIRun::Semaphore;
using SCIRun::Runnable;


template <class Var, class Iter>
class PatchDataThread : public Runnable {
public:  
  PatchDataThread(DataArchive& archive, 
		  Iter iter,
		  const string& varname,
		  int matnum,
		  const Patch* patch,
		  double time, Semaphore* sema) :
    archive_(archive),
    iter_(iter),
    name_(varname),
    mat_(matnum),
    patch_(patch),
    time_(time),
    sema_(sema){}

  void run() 
    {
      Var v; 
      archive_.query( v, name_, mat_, patch_, time_); 
      *iter_ = v; 
      sema_->up();
    }
      
private:

  PatchDataThread(){}

  DataArchive& archive_;
  Iter iter_;
  const string& name_;
  int mat_;
  const Patch *patch_;
  double time_;
  Semaphore *sema_;
};
  


} // end namespace Uintah
#endif
