#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include "MPM/MPM.h"
#include "Euler/Euler.h"

using namespace SCICore::Containers;
using namespace PSECore::Dataflow;
using namespace McQ;

#include <stdio.h>

extern "C" void initPackage(const clString& tcl) {
  packageDB.registerModule("McQ","SAMRAI","Euler",Euler::make,tcl+"/Euler.tcl");
  packageDB.registerModule("McQ","MPM","Simulation",MPM::make,tcl+"/MPM.tcl");
}
