
#include <Packages/Uintah/CCA/Components/Models/test/SimpleRxn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>

using namespace Uintah;
using namespace std;

SimpleRxn::SimpleRxn(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;

  massFraction = VarLabel::create("SimpleRxn::massFraction",
				  CCVariable<double>::getTypeDescription());
}

SimpleRxn::~SimpleRxn()
{
  if(mymatls && mymatls->removeReference())
    delete mymatls;
  VarLabel::destroy(massFraction);
}

void SimpleRxn::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup&)
{
  matl = sharedState->parseAndLookupMaterial(params, "material");
  params->require("rate", rate);

  vector<int> m(1);
  m[0] = matl->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();
 
  // determine the specific heat of that matl.
  Material* matl = sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
  if (mpm_matl){
    d_cv_0 = mpm_matl->getSpecificHeat();
  }
  if (ice_matl){
    d_cv_0 = ice_matl->getSpecificHeat();
  }
}

void SimpleRxn::scheduleInitialize(const LevelP&,
				   SchedulerP&)
{
  //schedule the init of massFraction...;
  // None necessary...
}
      
void SimpleRxn::scheduleComputeStableTimestep(SchedulerP&,
					      const LevelP&,
					      const ModelInfo*)
{
  // None necessary...
}
      
void SimpleRxn::scheduleMassExchange(SchedulerP& sched,
				     const LevelP& level,
				     const ModelInfo* mi)
{
  Task* t = scinew Task("SimpleRxn::massExchange",
			this, &SimpleRxn::massExchange, mi);
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  Ghost::GhostType  gn  = Ghost::None;
  t->requires(Task::OldDW, mi->density_CCLabel,    matl->thisMaterial(), gn);
  t->requires(Task::OldDW, mi->velocity_CCLabel,   matl->thisMaterial(), gn);
  t->requires(Task::OldDW, mi->temperature_CCLabel,matl->thisMaterial(), gn);
  t->requires(Task::OldDW, mi->delT_Label);
  sched->addTask(t, level->eachPatch(), mymatls);
}

void SimpleRxn::massExchange(const ProcessorGroup*, 
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const ModelInfo* mi)
{
#if 0
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
    new_dw->getModifiable(mass_source_0,     mi->mass_source_CCLabel, 
                       m0, patch);
    new_dw->getModifiable(momentum_source_0, mi->momentum_source_CCLabel,
			  m0, patch);
    new_dw->getModifiable(energy_source_0,   mi->energy_source_CCLabel,
			  m0, patch);

    CCVariable<double> mass_source_1;
    CCVariable<Vector> momentum_source_1;
    CCVariable<double> energy_source_1;
    new_dw->getModifiable(mass_source_1,     mi->mass_source_CCLabel, 
                      m1, patch);
    new_dw->getModifiable(momentum_source_1, mi->momentum_source_CCLabel,
			  m1, patch);
    new_dw->getModifiable(energy_source_1,   mi->energy_source_CCLabel,
			  m1, patch);

    constCCVariable<double> density_0;
    constCCVariable<Vector> vel_0;
    constCCVariable<double> temp_0;
    old_dw->get(density_0, mi->density_CCLabel,     m0, patch, Ghost::None, 0);
    old_dw->get(vel_0,     mi->velocity_CCLabel,    m0, patch, Ghost::None, 0);
    old_dw->get(temp_0,    mi->temperature_CCLabel, m0, patch, Ghost::None, 0);

    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();
    double tm = 0;
    double trate = rate*dt;
    if(trate > 1)
      trate=1;
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      double mass = density_0[*iter] * volume;
      double massx = mass*trate;
      mass_source_0[*iter] -= massx;
      mass_source_1[*iter] += massx;

      Vector momx = vel_0[*iter]*massx;
      momentum_source_0[*iter] -= momx;
      momentum_source_1[*iter] += momx;
      
      double energyx = temp_0[*iter] * massx * d_cv_0;
      energy_source_0[*iter] -= energyx;
      energy_source_1[*iter] += energyx;

      tm += massx;
    }
    cerr << "Total mass transferred: " << tm << '\n';
  }
#endif
}

void SimpleRxn::scheduleMomentumAndEnergyExchange(SchedulerP&,
				       const LevelP&,
				       const ModelInfo*)
{
  // None
}
