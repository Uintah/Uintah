#include "ThermalContactFactory.h"
#include "ThermalContact.h"
#include <SCICore/Malloc/Allocator.h>
#include <string>
using std::cerr;

using namespace Uintah::MPM;

ThermalContact* ThermalContactFactory::create(const ProblemSpecP& ps,
  SimulationStateP& d_sS)
{
   for( ProblemSpecP child = ps->findBlock("thermal_contact"); child != 0;
        child = child->findNextBlock("thermal_contact"))
   {
     return( scinew ThermalContact(child,d_sS) );
   }
   return 0;
}

// $Log$
// Revision 1.2  2000/06/20 04:14:30  tan
// WHen SerialMPM::d_thermalContactModel != NULL, heat conduction will be included
// in MPM algorithm.  The d_thermalContactModel is set by ThermalContactFactory
// according to the information in ProblemSpec from input file.
//
// Revision 1.1  2000/06/20 03:20:35  tan
// Added ThermalContactFactory class to interface with ProblemSpecification.
//
