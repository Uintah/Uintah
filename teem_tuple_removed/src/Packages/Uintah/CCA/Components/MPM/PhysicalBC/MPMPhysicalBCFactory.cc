#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

std::vector<MPMPhysicalBC*> MPMPhysicalBCFactory::mpmPhysicalBCs;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps)
{
  ProblemSpecP test = ps->findBlock("PhysicalBC");
   if (!test)     // bullet proofing
    throw ProblemSetupException("**ERROR** No <PhysicalBC> block in input file.");
   
   ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");
   
   for(ProblemSpecP child = current_ps->findBlock("force"); child != 0;
       child = child->findNextBlock("force") )
   {
      mpmPhysicalBCs.push_back(scinew ForceBC(child));
   }

   for(ProblemSpecP child = current_ps->findBlock("pressure"); child != 0;
       child = child->findNextBlock("pressure") )
   {
      mpmPhysicalBCs.push_back(scinew PressureBC(child));
   }

   for(ProblemSpecP child = current_ps->findBlock("crack"); child != 0;
       child = child->findNextBlock("crack") )
   {
      mpmPhysicalBCs.push_back(scinew CrackBC(child));
   }
}

void MPMPhysicalBCFactory::clean()
{
  for (int i = 0; i < static_cast<int>(mpmPhysicalBCs.size()); i++)
    delete mpmPhysicalBCs[i];
  mpmPhysicalBCs.clear();
}
