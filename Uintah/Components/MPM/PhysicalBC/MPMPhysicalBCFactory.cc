#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <Uintah/Components/MPM/PhysicalBC/ForceBC.h>
#include <Uintah/Components/MPM/PhysicalBC/CrackBC.h>
#include <Uintah/Interface/ProblemSpec.h>

using namespace std;
using namespace Uintah::MPM;

std::vector<MPMPhysicalBC*> MPMPhysicalBCFactory::mpmPhysicalBCs;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps)
{
   ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");
   
   for(ProblemSpecP child = current_ps->findBlock("force"); child != 0;
       child = child->findNextBlock("force") )
   {
      mpmPhysicalBCs.push_back(new ForceBC(child));
   }

   for(ProblemSpecP child = current_ps->findBlock("crack"); child != 0;
       child = child->findNextBlock("crack") )
   {
      mpmPhysicalBCs.push_back(new CrackBC(child));
   }
}

// $Log$
// Revision 1.3  2000/12/30 05:08:12  tan
// Fixed a problem concerning patch and ghost in fracture computations.
//
// Revision 1.2  2000/08/18 20:30:14  tan
// Fixed some bugs in SerialMPM, mainly in applyPhysicalBC.
//
// Revision 1.1  2000/08/07 00:43:45  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
