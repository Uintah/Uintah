#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/NormalForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/ArchesHeatFluxBC.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

std::vector<MPMPhysicalBC*> MPMPhysicalBCFactory::mpmPhysicalBCs;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps)
{
  ProblemSpecP test = ps->findBlock("PhysicalBC");
  if (test){

    ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");


    for(ProblemSpecP child = current_ps->findBlock("force"); child != 0;
        child = child->findNextBlock("force") ) {
       mpmPhysicalBCs.push_back(scinew ForceBC(child));
    }

    for(ProblemSpecP child = current_ps->findBlock("normal_force"); child != 0;
        child = child->findNextBlock("normal_force") ) {
       mpmPhysicalBCs.push_back(scinew NormalForceBC(child));
    }

    for(ProblemSpecP child = current_ps->findBlock("pressure"); child != 0;
        child = child->findNextBlock("pressure") ) {
       mpmPhysicalBCs.push_back(scinew PressureBC(child));
    }

    for(ProblemSpecP child = current_ps->findBlock("crack"); child != 0;
        child = child->findNextBlock("crack") ) {
       mpmPhysicalBCs.push_back(scinew CrackBC(child));
    }

    for(ProblemSpecP child = current_ps->findBlock("heat_flux"); child != 0;
        child = child->findNextBlock("heat_flux") ) {
       mpmPhysicalBCs.push_back(scinew HeatFluxBC(child));
    }
    for(ProblemSpecP child = current_ps->findBlock("arches_heat_flux"); 
        child != 0; child = child->findNextBlock("arches_heat_flux") ) {
       mpmPhysicalBCs.push_back(scinew ArchesHeatFluxBC(child));
    }
  }
}

void MPMPhysicalBCFactory::clean()
{
  for (int i = 0; i < static_cast<int>(mpmPhysicalBCs.size()); i++)
    delete mpmPhysicalBCs[i];
  mpmPhysicalBCs.clear();
}
