


#include <unistd.h>
#include <iostream>
#include <Core/Thread/Thread.h>
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/PIDL/PIDL.h>

using namespace sci_cca;
using namespace SCIRun;

int
main( int argc, char *argv[] )
{
  bool stop = argc == 1;
  PIDL::PIDL::initialize();

  ComponentImpl *cp = argc == 1 ? 0 : new ComponentImpl;
  
  //  delete cp1;
  if ( stop ) {
    Semaphore wait("main wait",0);
    wait.down();
  }

  //delete cp;
  cerr << "main done\n";

  // never reached
  return 0;
}

