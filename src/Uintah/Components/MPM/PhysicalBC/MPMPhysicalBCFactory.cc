#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <Uintah/Components/MPM/PhysicalBC/ForceBC.h>
#include <Uintah/Interface/ProblemSpec.h>

using namespace std;
using namespace Uintah::MPM;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps,
                              std::vector<MPMPhysicalBC*>& bcs)
{
   ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");
   
   for(ProblemSpecP child = current_ps->findBlock("force"); child != 0;
       child = child->findNextBlock("force") )
   {
      bcs.push_back(new ForceBC(child));
   }
}

// $Log$
// Revision 1.1  2000/08/07 00:43:45  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
