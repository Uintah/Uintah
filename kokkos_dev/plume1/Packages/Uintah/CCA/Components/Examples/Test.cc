#include <Packages/Uintah/CCA/Components/Examples/Test.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Test::Test(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  cout << "Instantiating Test" << endl;
  SVariableLabel = VarLabel::create("sole.double",
                                   SoleVariable<double>::getTypeDescription());

  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());

  switchLabel = VarLabel::create("switch.bool",
                                 SoleVariable<bool>::getTypeDescription());
 
}

Test::~Test()
{
  VarLabel::destroy(SVariableLabel);
  VarLabel::destroy(delt_label);
  VarLabel::destroy(switchLabel);
  
}

void Test::problemSetup(const ProblemSpecP& params, GridP& /*grid*/,
                        SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  matl = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(matl);
}
 
void Test::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Test::initialize);
  task->computes(SVariableLabel);

  sched->addTask(task, level->eachPatch(),sharedState_->allMaterials());
}
 
void Test::scheduleComputeStableTimestep(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Test::computeStableTimestep);
  task->computes(delt_label);
  sched->addTask(task, level->eachPatch(),sharedState_->allMaterials());
}

void
Test::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched,
                          int, int )
{
  Task* task = scinew Task("timeAdvance",
			   this, &Test::timeAdvance);
  task->requires(Task::OldDW, SVariableLabel);
  task->computes(SVariableLabel);
  sched->addTask(task, level->eachPatch(),sharedState_->allMaterials());
}

void
Test::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("switchTest",
			   this, &Test::switchTest);

  task->requires(Task::NewDW, SVariableLabel);
  task->computes(switchLabel);
  sched->addTask(task, level->eachPatch(),sharedState_->allMaterials());
}


void Test::computeStableTimestep(const ProcessorGroup* pg,
                                 const PatchSubset* /*patches*/,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{
  delt_ = 1.;
  new_dw->put(delt_vartype(delt_),delt_label);
}

void Test::initialize(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    //const Patch* patch = patches->get(p);
    
    SoleVariable<double> sdouble(1.1);
    new_dw->put(sdouble,SVariableLabel,getLevel(patches));

  }
}



void Test::timeAdvance(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    //const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      //int matl = matls->get(m);

      delt_vartype delt;
      old_dw->get(delt,delt_label,getLevel(patches));

      SoleVariable<double> sdouble;
      old_dw->get(sdouble,SVariableLabel,getLevel(patches));
      double total;
      total = sdouble + sdouble * delt;
      SoleVariable<double> sdouble_new(total);

      cout << "Test sdouble_new = " << sdouble_new << endl;

      new_dw->put(sdouble_new,SVariableLabel,getLevel(patches));
      
    }
  }
}

void Test::switchTest(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  SoleVariable<double> sdouble;
  new_dw->get(sdouble,SVariableLabel,getLevel(patches));

  bool sw = false;
  if (sdouble > 70.)
    sw = true;
  else
    sw = false;
  
  SoleVariable<bool> switch_condition(sw);
  new_dw->put(switch_condition,switchLabel,getLevel(patches));

}
