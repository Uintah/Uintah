#include "HeatConductionFactory.h"
#include "HeatConduction.h"
#include <Core/Malloc/Allocator.h>
#include <string>

using std::cerr;

using namespace Uintah;

HeatConduction* HeatConductionFactory::create(const ProblemSpecP& ps,
  SimulationStateP& d_sS)
{
  ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

   for( ProblemSpecP child = mpm_ps->findBlock("heat_conduction"); child != 0;
        child = child->findNextBlock("heat_conduction"))
   {
     return( scinew HeatConduction(child,d_sS) );
   }
   return 0;
}

