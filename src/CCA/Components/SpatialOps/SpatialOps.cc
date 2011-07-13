#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/CoalModels/ConstantModel.h>
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/SourceTerms/ConstSrcTerm.h>
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/SpatialOps/SpatialOpsMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Output.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Box.h>
#include <Core/Thread/Time.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <fstream>

using namespace std;

//===========================================================================

namespace Uintah {

SpatialOps::SpatialOps(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{

  d_fieldLabels = scinew ArchesLabel();//  scinew Fields();

  nofTimeSteps = 0;

  d_pi = acos(-1.0);
}

SpatialOps::~SpatialOps()
{
  delete d_fieldLabels;
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SpatialOps::problemSetup(const ProblemSpecP& params, 
                         const ProblemSpecP& materials_ps, 
                         GridP& grid, 
                         SimulationStateP& sharedState)
{ 
  d_sharedState = sharedState;
  d_doDQMOM = false; 

  // Input
  ProblemSpecP db = params->findBlock("CFD")->findBlock("SPATIALOPS");
  db->require("lambda", d_initlambda);  
  db->getWithDefault("temperature", d_initTemperature,298.0);
  ProblemSpecP time_db = db->findBlock("TimeIntegrator");
  time_db->getWithDefault("tOrder",d_tOrder,1); 

  // define a single material for now.
  SpatialOpsMaterial* mat = scinew SpatialOpsMaterial();
  sharedState->registerSpatialOpsMaterial(mat);

  //create a time integrator.
  d_timeIntegrator = scinew ExplicitTimeInt(d_fieldLabels);
  d_timeIntegrator->problemSetup(time_db);

  //set shared state in fields
  d_fieldLabels->setSharedState(sharedState);

}
//---------------------------------------------------------------------------
// Method: Schedule Initialize
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleInitialize(const LevelP& level,
                 SchedulerP& sched)
{
  Task* tsk = scinew Task("SpatialOps::actuallyInitialize", this, &SpatialOps::actuallyInitialize);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually Initialize
//---------------------------------------------------------------------------
void
SpatialOps::actuallyInitialize(const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* ,
                   DataWarehouse* new_dw)
{
}
//---------------------------------------------------------------------------
// Method: Schedule Compute Stable Timestep
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleComputeStableTimestep(const LevelP& level,
                        SchedulerP& sched)
{

  Task* tsk = scinew Task("SpatialOps::computeStableTimestep",
              this, &SpatialOps::computeStableTimestep);
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Compute Stable Time Step
//---------------------------------------------------------------------------
void 
SpatialOps::computeStableTimestep(const ProcessorGroup* ,
                    const PatchSubset* patches,
                        const MaterialSubset*,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  //double deltat = 100000000.0;
}
//---------------------------------------------------------------------------
// Method: Schedule Time Advance
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleTimeAdvance(const LevelP& level, 
                  SchedulerP& sched)
{
  // double time = d_sharedState->getElapsedTime();
  nofTimeSteps++;

  double start_time = Time::currentSeconds();

  for (int i = 0; i < d_tOrder; i++){

  }
  double end_time = Time::currentSeconds();
  cout << "Solution time = " << end_time - start_time << endl;
}
//---------------------------------------------------------------------------
// Method: Need Recompile
//---------------------------------------------------------------------------
bool SpatialOps::needRecompile(double time, double dt, 
                  const GridP& grid) 
{
 return d_recompile;
}
} //namespace Uintah
