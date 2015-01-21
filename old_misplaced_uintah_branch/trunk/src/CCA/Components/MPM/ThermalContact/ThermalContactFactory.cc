#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <CCA/Components/MPM/ThermalContact/STThermalContact.h>
#include <CCA/Components/MPM/ThermalContact/NullThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <string>
using std::cerr;

using namespace Uintah;

ThermalContact* ThermalContactFactory::create(const ProblemSpecP& ps,
					      SimulationStateP& d_sS, 
					      MPMLabel* lb,MPMFlags* flag)
{
   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

   for( ProblemSpecP child = mpm_ps->findBlock("thermal_contact"); child != 0;
	        child = child->findNextBlock("thermal_contact")) {
     return( scinew STThermalContact(child,d_sS,lb,flag) );
   }

   ProblemSpecP child; 
   return( scinew NullThermalContact(child,d_sS,lb,flag) );
}
