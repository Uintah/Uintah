#include "ThermalContactFactory.h"
#include "ThermalContact.h"
#include <Core/Malloc/Allocator.h>
#include <string>
using std::cerr;

using namespace Uintah;

ThermalContact* ThermalContactFactory::create(const ProblemSpecP& ps,
  SimulationStateP& d_sS)
{
   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

//   for( ProblemSpecP child = mpm_ps->findBlock("thermal_contact"); child != 0;
//        child = child->findNextBlock("thermal_contact"))
//   {
     ProblemSpecP child; 
     return( scinew ThermalContact(child,d_sS) );
//   }
//   return 0;
}

