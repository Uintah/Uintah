



#include <unistd.h>
#include <iostream>
#include <Core/Thread/Thread.h>
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/BuilderImpl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/Component/PIDL/PIDL.h>

using namespace sci_cca;
using namespace SCIRun;

int
main( int argc, char *argv[] )
{
  bool server = argc== 1;

  if ( ! CCA::init( argc, argv ) ) {
    cerr << "cca init error\n";
    return 1;
  }

  cerr << "main cont.\n";

  Component b;
  if ( !server ) {
    b = new BuilderImpl;
    CCA::init( b );
    cerr << "CCA::init done\n";
  } 
  else {
    Semaphore wait("main wait",0);
    wait.down();
  }

  cerr << "main continue\n";

  //  b->setServices(0);

  cerr << "main done\n";

  return 0;
}

