#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

std::vector<MPMPhysicalBC*> MPMPhysicalBCFactory::mpmPhysicalBCs;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps)
{
   ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");
   
   for(ProblemSpecP child = current_ps->findBlock("force"); child != 0;
       child = child->findNextBlock("force") )
   {
      mpmPhysicalBCs.push_back(scinew ForceBC(child));
   }

   for(ProblemSpecP child = current_ps->findBlock("crack"); child != 0;
       child = child->findNextBlock("crack") )
   {
      mpmPhysicalBCs.push_back(scinew CrackBC(child));
   }
}

