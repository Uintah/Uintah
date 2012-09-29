/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/SwitchingCriteria/SteadyState.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <string>
#include <iostream>

using namespace std;
using namespace Uintah;
static DebugStream dbg("SWITCHER", false);

SteadyState::SteadyState(ProblemSpecP& ps)
{
  ps->require("material", d_material);
  ps->require("num_steps", d_numSteps);

  proc0cout << "material = " << d_material << endl;
  proc0cout << "num_steps  = " << d_numSteps << endl;

  heatRate_CCLabel = 
    VarLabel::create("heatRate_CC",CCVariable<double>::getTypeDescription());

  heatFluxSumLabel = 
    VarLabel::create("heatFluxSum",sum_vartype::getTypeDescription() );

  heatFluxSumTimeDerivativeLabel = 
    VarLabel::create("heatFluxSumTimeDerivative", 
                     sum_vartype::getTypeDescription() );
}

SteadyState::~SteadyState()
{
  VarLabel::destroy(heatRate_CCLabel);
  VarLabel::destroy(heatFluxSumLabel);
  VarLabel::destroy(heatFluxSumTimeDerivativeLabel);
}

void SteadyState::problemSetup(const ProblemSpecP& ps, 
                               const ProblemSpecP& restart_prob_spec, 
                               SimulationStateP& state)
{
  d_sharedState = state;
}

void SteadyState::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{

  Task* t = scinew Task("SteadyState::actuallyInitialize",
                        this, &SteadyState::initialize);
  t->computes(heatFluxSumLabel);
  t->computes(heatFluxSumTimeDerivativeLabel);
  t->computes(d_sharedState->get_switch_label());

  sched->addTask(t, level->eachPatch(), d_sharedState->allMaterials());

}

void SteadyState::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse*,
                             DataWarehouse* new_dw)
{
  proc0cout << "Initializing heatFluxSum and heatFluxSumTimeDerivative" << endl;
 
  new_dw->put(max_vartype(0.0), heatFluxSumLabel);
  new_dw->put(max_vartype(0.0), heatFluxSumTimeDerivativeLabel);
  new_dw->put(max_vartype(0.0),d_sharedState->get_switch_label());

}

void SteadyState::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,dbg,"Switching Criteria:SteadyState::scheduleSwitchTest");
  
  Task* t = scinew Task("switchTest", this, &SteadyState::switchTest);

  MaterialSubset* container = scinew MaterialSubset();

  container->add(d_material);
  container->addReference();

  t->requires(Task::NewDW, heatRate_CCLabel,container,Ghost::None);
  t->requires(Task::OldDW, heatFluxSumLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->computes(heatFluxSumLabel);
  t->computes(heatFluxSumTimeDerivativeLabel);
  t->computes(d_sharedState->get_switch_label());

  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());

  scheduleDummy(level,sched);
}

void SteadyState::switchTest(const ProcessorGroup* group,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  double sw = 0;
  double heatFluxSum = 0;


  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);  
    printTask(patches, patch,dbg,"Doing Switching Criteria:SimpleBurnCriteria::switchTest");
    
    constCCVariable<double> heatFlux;
    new_dw->get(heatFlux, heatRate_CCLabel,0,patch,Ghost::None,0);
    
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      heatFluxSum += heatFlux[*iter];
    }   
  }

  new_dw->put(max_vartype(heatFluxSum),heatFluxSumLabel);
  proc0cout << "heatFluxSum = " << heatFluxSum << endl;

  max_vartype oldHeatFluxSum;
  old_dw->get(oldHeatFluxSum,heatFluxSumLabel);
  proc0cout << "oldHeatFluxSum = " << oldHeatFluxSum << endl;

  delt_vartype delT;
  old_dw->get(delT,d_sharedState->get_delt_label(),getLevel(patches));

  double dH_dt = (heatFluxSum - oldHeatFluxSum)/delT;
  max_vartype heatFluxSumTimeDerivative(dH_dt);
  proc0cout << "heatFluxSumTimeDerivative = " << heatFluxSumTimeDerivative << endl;

  new_dw->put(heatFluxSumTimeDerivative,heatFluxSumTimeDerivativeLabel);

  max_vartype switch_condition(sw);

  const Level* allLevels = 0;
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),allLevels);

}


void SteadyState::scheduleDummy(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("SteadyState::dummy", this, &SteadyState::dummy);
  t->requires(Task::OldDW,d_sharedState->get_switch_label(),level.get_rep());
  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());
}

void SteadyState::dummy(const ProcessorGroup* group,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  max_vartype old_sw(1.23);
  old_dw->get(old_sw,d_sharedState->get_switch_label());
  proc0cout << "old_sw = " << old_sw << endl;
}
