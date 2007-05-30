#include <CCA/Components/SwitchingCriteria/TimestepNumber.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <string>
#include <iostream>

using namespace std;

using namespace Uintah;

TimestepNumber::TimestepNumber(ProblemSpecP& ps)
{
  ps->require("timestep",d_timestep);
  if (Parallel::getMPIRank() == 0) {
    cout << "timestep = " << d_timestep << endl;
  }
}

TimestepNumber::~TimestepNumber()
{
}

void TimestepNumber::problemSetup(const ProblemSpecP& ps, 
                                  const ProblemSpecP& restart_prob_spec, 
                                  SimulationStateP& state)
{
  d_sharedState = state;
}

void TimestepNumber::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("switchTest", this, &TimestepNumber::switchTest);

  t->computes(d_sharedState->get_switch_label(), level.get_rep());
  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());
}

void TimestepNumber::switchTest(const ProcessorGroup* group,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  double sw = 0;

  unsigned int time_step = d_sharedState->getCurrentTopLevelTimeStep();
  if (time_step == d_timestep)
    sw = 1;
  else
    sw = 0;

  max_vartype switch_condition(sw);
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));
}
