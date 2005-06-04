#include <Packages/Uintah/CCA/Components/SimulationController/SimulationControllerFactory.h>
#include <Packages/Uintah/CCA/Components/SimulationController/MultipleSimulationController.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimpleSimulationController.h>
#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

SimulationController* SimulationControllerFactory::create(ProblemSpecP& ps, 
                                                          const ProcessorGroup* world)
{
  SimulationController* ctl = 0;
  string simulation_ctl = "";
  
  ps->get("SimulationController",simulation_ctl);

  if(simulation_ctl == "AMR" || simulation_ctl == "amr"){
    AMRSimulationController* amr_ctl 
      = scinew AMRSimulationController(world,true);
    ctl = amr_ctl;
  } else if(simulation_ctl == "Multiple" || simulation_ctl == "multiple") {
    MultipleSimulationController* mult_ctl 
      = scinew MultipleSimulationController(world);
    ctl = mult_ctl;
  } else if(simulation_ctl == "Simple" || simulation_ctl == "simple") {
    SimpleSimulationController* sim_ctl
      = scinew SimpleSimulationController(world);
    ctl = sim_ctl;
  } else {
    ctl = scinew SimpleSimulationController(world);   
    cerr << "Unknown simulation controller, using SimpleSimulationController"
         << endl;
  }
  
  return ctl;

}
