#include "ThermalContactFactory.h"
#include "ThermalContact.h"
#include <SCICore/Malloc/Allocator.h>
#include <string>
using std::cerr;

using namespace Uintah::MPM;

ThermalContact* ThermalContactFactory::create(ProblemSpecP& ps)
{
   for( ProblemSpecP child = ps->findBlock("thermal_contact"); child != 0;
        child = child->findNextBlock("thermal_contact"))
   {
     return( scinew ThermalContact() );
   }
   return 0;
}

// $Log$
// Revision 1.1  2000/06/20 03:20:35  tan
// Added ThermalContactFactory class to interface with ProblemSpecification.
//
