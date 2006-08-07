#include <Packages/Uintah/CCA/Components/Regridder/RegridderFactory.h>
#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

using namespace Uintah;

RegridderCommon* RegridderFactory::create(ProblemSpecP& ps, 
                                          const ProcessorGroup* world)
{
  RegridderCommon* regrid = 0;
  
  ProblemSpecP amr = ps->findBlock("AMR");	
  ProblemSpecP reg_ps = amr->findBlock("Regridder");
  if (reg_ps) {
    // only instantiate if there is a Regridder section.  If 
    // no type specified, call it 'Hierarchical'
    string regridder = "Hierarchical";
    reg_ps->get("type",regridder);

    if(regridder == "Hierarchical") {
      regrid = new HierarchicalRegridder(world);
    } else if(regridder == "BNR") {
      regrid = new BNRRegridder(world);
    } else
      regrid = 0;
  }

  return regrid;

}
