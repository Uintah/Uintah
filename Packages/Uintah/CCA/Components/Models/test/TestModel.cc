
#include <Packages/Uintah/CCA/Components/Models/test/TestModel.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <iostream>

using namespace Uintah;
using namespace std;

TestModel::TestModel(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
}

TestModel::~TestModel()
{
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void TestModel::problemSetup(GridP& grid, SimulationStateP& sharedState,
			     ModelSetup& setup)
{
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");
  params->require("rate", rate);

  vector<int> m(2);
  m[0] = matl0->getDWIndex();
  m[1] = matl1->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();
}
      
void TestModel::scheduleInitialize(const LevelP&,
				   SchedulerP&)
{
  // None necessary...
}
      
void TestModel::scheduleComputeStableTimestep(SchedulerP&,
					      const LevelP& level,
					      const ModelInfo*)
{
  // None necessary...
}
      
void TestModel::scheduleMassExchange(SchedulerP& sched,
				     const LevelP& level,
				     const ModelInfo* mi)
{
  Task* t = scinew Task("TestModel::massExchange",
			this, &TestModel::massExchange, mi);
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->requires(Task::OldDW, mi->density_CCLabel, matl0->thisMaterial(),
	      Ghost::None);
  t->requires(Task::OldDW, mi->velocity_CCLabel, matl0->thisMaterial(),
	      Ghost::None);
  t->requires(Task::OldDW, mi->temperature_CCLabel, matl0->thisMaterial(),
	      Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label);
  sched->addTask(t, level->eachPatch(), mymatls);
}

void TestModel::massExchange(const ProcessorGroup*, 
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  double dt = delT;
  ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    CCVariable<double> mass_source_0;
    CCVariable<Vector> momentum_source_0;
    CCVariable<double> energy_source_0;
    new_dw->getModifiable(mass_source_0, mi->mass_source_CCLabel, m0, patch);
    new_dw->getModifiable(momentum_source_0, mi->momentum_source_CCLabel,
			  m0, patch);
    new_dw->getModifiable(energy_source_0, mi->energy_source_CCLabel,
			  m0, patch);

    CCVariable<double> mass_source_1;
    CCVariable<Vector> momentum_source_1;
    CCVariable<double> energy_source_1;
    new_dw->getModifiable(mass_source_1, mi->mass_source_CCLabel, m1, patch);
    new_dw->getModifiable(momentum_source_1, mi->momentum_source_CCLabel,
			  m1, patch);
    new_dw->getModifiable(energy_source_1, mi->energy_source_CCLabel,
			  m1, patch);

    constCCVariable<double> density_0;
    constCCVariable<Vector> vel_0;
    constCCVariable<double> temp_0;
    old_dw->get(density_0, mi->density_CCLabel, m0, patch, Ghost::None, 0);
    old_dw->get(vel_0, mi->velocity_CCLabel, m0, patch, Ghost::None, 0);
    old_dw->get(temp_0, mi->temperature_CCLabel, m0, patch, Ghost::None, 0);

    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();
    double tm = 0;
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      double mass = density_0[*iter] * volume;
      double massx = mass*rate*dt;
      mass_source_0[*iter] -= massx;
      mass_source_1[*iter] += massx;

      Vector momx = vel_0[*iter]*massx*rate;
      momentum_source_0[*iter] -= momx;
      momentum_source_1[*iter] += momx;

      double energyx = temp_0[*iter]*massx*rate;
      energy_source_0[*iter] -= energyx;
      energy_source_1[*iter] += energyx;

      tm += massx;
    }
    cerr << "Total mass transferred: " << tm << '\n';
  }
}

void TestModel::scheduleMomentumAndEnergyExchange(SchedulerP&,
				       const LevelP& level,
				       const ModelInfo*)
{
  // None
}
