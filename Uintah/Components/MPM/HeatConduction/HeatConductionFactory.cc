#include "HeatConductionFactory.h"
#include "HeatConduction.h"
#include <SCICore/Malloc/Allocator.h>
#include <string>

using std::cerr;

using namespace Uintah::MPM;

HeatConduction* HeatConductionFactory::create(const ProblemSpecP& ps,
  SimulationStateP& d_sS)
{
   for( ProblemSpecP child = ps->findBlock("heat_conduction"); child != 0;
        child = child->findNextBlock("heat_conduction"))
   {
     return( scinew HeatConduction(child,d_sS) );
   }
   return 0;
}

// $Log$
// Revision 1.1  2000/06/20 17:59:37  tan
// Heat Conduction model created to move heat conduction part of code from MPM.
// Thus make MPM clean and easy to maintain.
//
