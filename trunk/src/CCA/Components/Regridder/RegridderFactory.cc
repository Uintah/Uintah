#include <CCA/Components/Regridder/RegridderFactory.h>
#include <CCA/Components/Regridder/HierarchicalRegridder.h>
#include <CCA/Components/Regridder/BNRRegridder.h>
#include <Core/Parallel/ProcessorGroup.h>

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

    if (world->myrank() == 0)
      cout << "Using Regridder " << regridder << endl;


    if(regridder == "Hierarchical") {
      regrid = scinew HierarchicalRegridder(world);
    } else if(regridder == "BNR") {
      regrid = scinew BNRRegridder(world);
    } else
      regrid = 0;
  }

  return regrid;

}
