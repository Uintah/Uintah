
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/Builder.h>
#include <testprograms/Component/framework/ConnectionServicesImpl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/Component/PIDL/PIDL.h>

#include <unistd.h>
#include <iostream>

using namespace sci_cca;
using namespace SCIRun;
using namespace std;

int
main( int argc, char *argv[] )
{
  if ( ! CCA::init( argc, argv ) ) {
    cerr << "cca init error\n";
    return 1;
  }

  cerr << "main cont.\n";

  Component b = new Builder();
  CCA::init( b );

  cerr << "done\n";
  CCA::done();

  cerr << "main done\n";

  return 0;
}

