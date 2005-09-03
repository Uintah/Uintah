#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SteadyState.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <string>
#include <iostream>

using namespace std;

using namespace Uintah;

SteadyState::SteadyState(ProblemSpecP& ps)
{
  ps->require("material", d_material);
  ps->require("num_steps", d_numSteps);

  cout << "material = " << d_material << endl;
  cout << "num_steps  = " << d_numSteps << endl;

  heatFlux_CCLabel = 
    VarLabel::create("heatFlux_CC",CCVariable<double>::getTypeDescription());

  heatFluxSumLabel = 
    VarLabel::create("heatFluxSum",sum_vartype::getTypeDescription() );

  heatFluxSumTimeDerivativeLabel = 
    VarLabel::create("heatFluxSumTimeDerivative", 
                     sum_vartype::getTypeDescription() );
}

SteadyState::~SteadyState()
{
  VarLabel::destroy(heatFlux_CCLabel);
  VarLabel::destroy(heatFluxSumLabel);
  VarLabel::destroy(heatFluxSumTimeDerivativeLabel);
}

void SteadyState::problemSetup(const ProblemSpecP& ps, 
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
  cout << "Initializing heatFluxSum and heatFluxSumTimeDerivative" << endl;
  new_dw->put(max_vartype(0.0), heatFluxSumLabel);
  new_dw->put(max_vartype(0.0), heatFluxSumTimeDerivativeLabel);
  new_dw->put(max_vartype(0.0),d_sharedState->get_switch_label());

}

void SteadyState::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("switchTest", this, &SteadyState::switchTest);

  MaterialSubset* container = scinew MaterialSubset();

  container->add(d_material);
  container->addReference();

  t->requires(Task::NewDW, heatFlux_CCLabel,container,Ghost::None);
  t->requires(Task::OldDW, heatFluxSumLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->computes(heatFluxSumLabel);
  t->computes(heatFluxSumTimeDerivativeLabel);
  t->computes(d_sharedState->get_switch_label(), level.get_rep());

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
    
    constCCVariable<double> heatFlux;
    new_dw->get(heatFlux, heatFlux_CCLabel,0,patch,Ghost::None,0);
    
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      heatFluxSum += heatFlux[*iter];
    }   
  }

  new_dw->put(max_vartype(heatFluxSum),heatFluxSumLabel);
  cout << "heatFluxSum = " << heatFluxSum << endl;

  max_vartype oldHeatFluxSum;
  old_dw->get(oldHeatFluxSum,heatFluxSumLabel);

  cout << "oldHeatFluxSum = " << oldHeatFluxSum << endl;

  delt_vartype delT;
  old_dw->get(delT,d_sharedState->get_delt_label(),getLevel(patches));

  double dH_dt = (heatFluxSum - oldHeatFluxSum)/delT;
  max_vartype heatFluxSumTimeDerivative(dH_dt);

  cout << "heatFluxSumTimeDerivative = " << heatFluxSumTimeDerivative << endl;

  new_dw->put(heatFluxSumTimeDerivative,heatFluxSumTimeDerivativeLabel);

  max_vartype switch_condition(sw);
  new_dw->put(switch_condition,d_sharedState->get_switch_label());

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
  cout << "old_sw = " << old_sw << endl;
}
