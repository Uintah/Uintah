
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/Builders/Builder.h>
#include <testprograms/Component/framework/BuilderServicesImpl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/PIDL/PIDL.h>

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

  Builder * b = new Builder();
  Component::pointer bc(b);
  CCA::init( bc, "Builder" );

  b->ui(); // infinite ui loop (until ui quits)

  CCA::done();
  return 0;
}

