//----- MomentumSolver.cc ----------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RHSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/OdtClosure.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>


using namespace Uintah;
using namespace std;
#include <Packages/Uintah/CCA/Components/Arches/fortran/computeVel_fort.h>

//****************************************************************************
// Default constructor for MomentumSolver
//****************************************************************************
MomentumSolver::
MomentumSolver(const ArchesLabel* label, const MPMArchesLabel* MAlb,
	       TurbulenceModel* turb_model,
	       BoundaryCondition* bndry_cond,
	       PhysicalConstants* physConst) : 
                                   d_lab(label), d_MAlab(MAlb),
                                   d_turbModel(turb_model), 
                                   d_boundaryCondition(bndry_cond),
				   d_physicalConsts(physConst)
{
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
MomentumSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");

  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"upwind");
    if (conv_scheme == "upwind") d_central = false;
      else if (conv_scheme == "central") d_central = true;
	else throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  db->getWithDefault("pressure_correction",d_pressure_correction,false);
  db->getWithDefault("filter_divergence_constraint",d_filter_divergence_constraint,false);

  d_source = scinew Source(d_turbModel, d_physicalConsts);
  if (d_doMMS)
	  d_source->problemSetup(db);

  d_rhsSolver = scinew RHSSolver();

  d_mixedModel=d_turbModel->getMixedModel();
}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solve(SchedulerP& sched,
		      const PatchSet* patches,
		      const MaterialSet* matls,
		      const TimeIntegratorLabel* timelabels,
		      int index)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrix(sched, patches, matls, timelabels, index);
    

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrix(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls,
		    			const TimeIntegratorLabel* timelabels,
					int index)
{
  string taskname =  "MomentumSolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
			  this, &MomentumSolver::buildLinearMatrix,
			  timelabels, index);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, timelabels->pressure_out,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_uVelocitySPBCLabel);

    break;

  case Arches::YDIR:

    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_vVelocitySPBCLabel);

    break;

  case Arches::ZDIR:

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_wVelocitySPBCLabel);

    break;

  default:

    throw InvalidValue("Invalid index in MomentumSolver", __FILE__, __LINE__);
    
  }
	
  sched->addTask(tsk, patches, matls);


}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void 
MomentumSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const PatchSubset* patches,
				  const MaterialSubset* /*matls*/,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
		    		  const TimeIntegratorLabel* timelabels,
				  int index)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // Get the required data
    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    new_dw->get(constVelocityVars.pressure, timelabels->pressure_out, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_MAlab) {
      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		      matlIndex, patch, 
		      Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    
    switch (index) {

    case Arches::XDIR:

      new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelocitySPBCLabel,
		             matlIndex, patch);
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		  matlIndex, patch);

      break;

    case Arches::YDIR:

      new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelocitySPBCLabel,
		             matlIndex, patch);
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		  matlIndex, patch);

      break;

    case Arches::ZDIR:

      new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel,
		             matlIndex, patch);
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		  matlIndex, patch);

      break;

    default:

      throw InvalidValue("Invalid index in MomentumSolver", __FILE__, __LINE__);

    }

    
    // Actual compute operations

    if (d_MAlab) {
      
      d_boundaryCondition->calculateVelocityPred_mm(pc, patch, 
						    delta_t, index, cellinfo,
						    &velocityVars,
						    &constVelocityVars);

    }
    else {
    
      d_rhsSolver->calculateVelocity(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars,
				      &constVelocityVars);

      if (d_boundaryCondition->getIntrusionBC())
	d_boundaryCondition->calculateIntrusionVel(pc, patch,
						   index, cellinfo,
						   &velocityVars,
						   &constVelocityVars);
    }
    if ((d_boundaryCondition->getOutletBC())||(d_boundaryCondition->getPressureBC()))
    d_boundaryCondition->addPresGradVelocityOutletBC(pc, patch, index, cellinfo,
						     delta_t, &velocityVars,
						     &constVelocityVars);
    if ((d_boundaryCondition->getOutletBC())||(d_boundaryCondition->getPressureBC()))
    d_boundaryCondition->velocityPressureBC(pc, patch, index,
					    &velocityVars, &constVelocityVars);

  }
}

//****************************************************************************
// Schedule calculation of hat velocities
//****************************************************************************
void MomentumSolver::solveVelHat(const LevelP& level,
				 SchedulerP& sched,
		    		 const TimeIntegratorLabel* timelabels)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));

  sched_buildLinearMatrixVelHat(sched, patches, matls, timelabels);

}



// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixVelHat(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls,
					  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "MomentumSolver::BuildCoeffVelHat" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, 
			  this, &MomentumSolver::buildLinearMatrixVelHat,
			  timelabels);

  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  if (timelabels->multiple_steps)
    tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  else
    tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) {
    old_values_dw = parent_old_dw;
    tsk->requires(old_values_dw, d_lab->d_densityCPLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }
  else {
    old_values_dw = Task::NewDW;
    tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }
  tsk->requires(old_values_dw, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(old_values_dw, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(old_values_dw, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);

  if (d_pressure_correction)
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, timelabels->pressure_guess, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::NewDW, timelabels->pressure_guess, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // required for computing div constraint
//#ifdef divergenceconstraint
  if (timelabels->multiple_steps)
    tsk->requires(Task::NewDW, d_lab->d_scalarTempLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::OldDW, d_lab->d_divConstraintLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefSrcLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
//#endif

  if ((dynamic_cast<const OdtClosure*>(d_turbModel))||d_mixedModel) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel,
		    d_lab->d_tensorMatl,Task::OutOfDomain,
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
   else 
    tsk->requires(Task::NewDW, d_lab->d_stressTensorCompLabel,
                  d_lab->d_tensorMatl,Task::OutOfDomain,
                  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);
    
//#ifdef divergenceconstraint
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->computes(d_lab->d_divConstraintLabel);
  else
    tsk->modifies(d_lab->d_divConstraintLabel);
//#endif

  sched->addTask(tsk, patches, matls);
}




// ***********************************************************************
// Actual build of linear matrices for momentum components
// ***********************************************************************

void 
MomentumSolver::buildLinearMatrixVelHat(const ProcessorGroup* pc,
					const PatchSubset* patches,
					const MaterialSubset* /*matls*/,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw,
					const TimeIntegratorLabel* timelabels)
{
  TAU_PROFILE_TIMER(input, "Input", "[MomSolver::buildMVelHatPred::input]" , TAU_USER);
  TAU_PROFILE_TIMER(inputcell, "Inputcell", "[MomSolver::buildMVelHatPred::inputcell]" , TAU_USER);
  TAU_PROFILE_TIMER(compute, "Compute", "[MomSolver::buildMVelHatPred::compute]" , TAU_USER);
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;
  double time=d_lab->d_sharedState->getElapsedTime();
	  
  for (int p = 0; p < patches->size(); p++) {
  TAU_PROFILE_START(input);

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    if (timelabels->multiple_steps)
      new_dw->get(constVelocityVars.density, d_lab->d_densityTempLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    else
      old_dw->get(constVelocityVars.density, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) {
      old_values_dw = parent_old_dw;
      old_values_dw->get(constVelocityVars.old_density, d_lab->d_densityCPLabel,
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    else {
      old_values_dw = new_dw;
      old_values_dw->get(constVelocityVars.old_density, d_lab->d_densityTempLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    old_values_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constVelocityVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);

    if (d_pressure_correction)
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      old_dw->get(constVelocityVars.pressure, timelabels->pressure_guess, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      new_dw->get(constVelocityVars.pressure, timelabels->pressure_guess, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

//#ifdef divergenceconstraint
    if (timelabels->multiple_steps)
      new_dw->get(constVelocityVars.scalar, d_lab->d_scalarTempLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      old_dw->get(constVelocityVars.scalar, d_lab->d_scalarSPLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    constCCVariable<double> old_divergence;
    old_dw->get(old_divergence, d_lab->d_divConstraintLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.drhodf, d_lab->d_drhodfCPLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constVelocityVars.scalarDiffusionCoeff[ii],
		  d_lab->d_scalDiffCoefLabel,
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.scalarDiffNonlinearSrc, 
	        d_lab->d_scalDiffCoefSrcLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      new_dw->allocateAndPut(velocityVars.divergence,
			     d_lab->d_divConstraintLabel, matlIndex, patch);
    else
      new_dw->getModifiable(velocityVars.divergence,
			    d_lab->d_divConstraintLabel, matlIndex, patch);
    velocityVars.divergence.initialize(0.0);
//#endif

  TAU_PROFILE_STOP(input);
  TAU_PROFILE_START(inputcell);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

  TAU_PROFILE_STOP(inputcell);
  TAU_PROFILE_START(input);

    for(int index = 1; index <= Arches::NDIM; ++index) {

      // get multimaterial momentum source terms
      // get velocities for MPMArches with extra ghost cells

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->get(constVelocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->get(constVelocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;

	case Arches::YDIR:

	  new_dw->get(constVelocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->get(constVelocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	case Arches::ZDIR:

	  new_dw->get(constVelocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->get(constVelocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocateTemporary(velocityVars.uVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(velocityVars.uVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocateTemporary(velocityVars.vVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(velocityVars.vVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocateTemporary(velocityVars.wVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(velocityVars.wVelocityConvectCoeff[ii],  patch);
	  break;
	default:
	  throw InvalidValue("invalid index for velocity in MomentumSolver", __FILE__, __LINE__);
	}
      }
  TAU_PROFILE_STOP(input);
  TAU_PROFILE_START(compute);

      // Calculate Velocity Coeffs :
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 

      d_discretize->calculateVelocityCoeff(pc, patch, 
					   delta_t, index, d_central, 
					   cellinfo, &velocityVars,
					   &constVelocityVars);

      // Calculate Velocity source
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      // get data
      // allocate
      
      switch(index) {

      case Arches::XDIR:

	new_dw->allocateTemporary(velocityVars.uVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.uVelNonlinearSrc,  patch);
	new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,
			       matlIndex, patch);
	velocityVars.uVelRhoHat.copy(constVelocityVars.old_uVelocity,
				     velocityVars.uVelRhoHat.getLowIndex(),
				     velocityVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(velocityVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
	new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,
			       matlIndex, patch);
	velocityVars.vVelRhoHat.copy(constVelocityVars.old_vVelocity,
				     velocityVars.vVelRhoHat.getLowIndex(),
				     velocityVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(velocityVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);
	new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,
			       matlIndex, patch);
	velocityVars.wVelRhoHat.copy(constVelocityVars.old_wVelocity,
				     velocityVars.wVelRhoHat.getLowIndex(),
				     velocityVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in MomentumSolver for calcVelSrc", __FILE__, __LINE__);

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &velocityVars,
					&constVelocityVars);
      if (d_doMMS)
          d_source->calculateVelMMSource(pc, patch, 
					delta_t, time, index,
					cellinfo, &velocityVars,
					&constVelocityVars);



      // for scalesimilarity model add stress tensor to the source of velocity eqn.
      if ((dynamic_cast<const OdtClosure*>(d_turbModel))||d_mixedModel) {
        StencilMatrix<constCCVariable<double> > stressTensor; //9 point tensor
        if (timelabels->integrator_step_number == 
            TimeIntegratorStepNumber::First)
	  for (int ii = 0; ii < d_lab->d_tensorMatl->size(); ii++) {
	    old_dw->get(stressTensor[ii], 
			d_lab->d_stressTensorCompLabel, ii, patch,
			Ghost::AroundCells, Arches::ONEGHOSTCELL);
	  }
	else
	  for (int ii = 0; ii < d_lab->d_tensorMatl->size(); ii++) {
	    new_dw->get(stressTensor[ii], 
			d_lab->d_stressTensorCompLabel, ii, patch,
			Ghost::AroundCells, Arches::ONEGHOSTCELL);
	  }

	IntVector indexLow = patch->getCellFORTLowIndex();
	IntVector indexHigh = patch->getCellFORTHighIndex();
	
	// set density for the whole domain


	      // Store current cell
	double sue, suw, sun, sus, sut, sub;
	switch (index) {
	case Arches::XDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             (stressTensor[0])[currCell];
		suw = cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             (stressTensor[0])[prevXCell];
		sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[1])[currCell]+
			      (stressTensor[1])[prevXCell]+
			      (stressTensor[1])[IntVector(colX,colY+1,colZ)]+
			      (stressTensor[1])[IntVector(colX-1,colY+1,colZ)]);
		sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[1])[currCell]+
			      (stressTensor[1])[prevXCell]+
			      (stressTensor[1])[IntVector(colX,colY-1,colZ)]+
			      (stressTensor[1])[IntVector(colX-1,colY-1,colZ)]);
		sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		             ((stressTensor[2])[currCell]+
			      (stressTensor[2])[prevXCell]+
			      (stressTensor[2])[IntVector(colX,colY,colZ+1)]+
			      (stressTensor[2])[IntVector(colX-1,colY,colZ+1)]);
		sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		             ((stressTensor[2])[currCell]+
			      (stressTensor[2])[prevXCell]+
			      (stressTensor[2])[IntVector(colX,colY,colZ-1)]+
			      (stressTensor[2])[IntVector(colX-1,colY,colZ-1)]);
		velocityVars.uVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	case Arches::YDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		  ((stressTensor[3])[currCell]+
		   (stressTensor[3])[prevYCell]+
		   (stressTensor[3])[IntVector(colX+1,colY,colZ)]+
		   (stressTensor[3])[IntVector(colX+1,colY-1,colZ)]);
		suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		  ((stressTensor[3])[currCell]+
		   (stressTensor[3])[prevYCell]+
		   (stressTensor[3])[IntVector(colX-1,colY,colZ)]+
		   (stressTensor[3])[IntVector(colX-1,colY-1,colZ)]);
		sun = cellinfo->sew[colX]*cellinfo->stb[colZ]*
		  (stressTensor[4])[currCell];
		sus = cellinfo->sew[colX]*cellinfo->stb[colZ]*
		  (stressTensor[4])[prevYCell];
		sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		  ((stressTensor[5])[currCell]+
		   (stressTensor[5])[prevYCell]+
		   (stressTensor[5])[IntVector(colX,colY,colZ+1)]+
		   (stressTensor[5])[IntVector(colX,colY-1,colZ+1)]);
		sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		  ((stressTensor[5])[currCell]+
		   (stressTensor[5])[prevYCell]+
		   (stressTensor[5])[IntVector(colX,colY,colZ-1)]+
		   (stressTensor[5])[IntVector(colX,colY-1,colZ-1)]);
		velocityVars.vVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	case Arches::ZDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             ((stressTensor[6])[currCell]+
			      (stressTensor[6])[prevZCell]+
			      (stressTensor[6])[IntVector(colX+1,colY,colZ)]+
			      (stressTensor[6])[IntVector(colX+1,colY,colZ-1)]);
		suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             ((stressTensor[6])[currCell]+
			      (stressTensor[6])[prevZCell]+
			      (stressTensor[6])[IntVector(colX-1,colY,colZ)]+
			      (stressTensor[6])[IntVector(colX-1,colY,colZ-1)]);
		sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[7])[currCell]+
			      (stressTensor[7])[prevZCell]+
			      (stressTensor[7])[IntVector(colX,colY+1,colZ)]+
			      (stressTensor[7])[IntVector(colX,colY+1,colZ-1)]);
		sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[7])[currCell]+
			      (stressTensor[7])[prevZCell]+
			      (stressTensor[7])[IntVector(colX,colY-1,colZ)]+
			      (stressTensor[7])[IntVector(colX,colY-1,colZ-1)]);
		sut = cellinfo->sew[colX]*cellinfo->sns[colY]*
		             (stressTensor[8])[currCell];
		sub = cellinfo->sew[colX]*cellinfo->sns[colY]*
		             (stressTensor[8])[prevZCell];
		velocityVars.wVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	default:
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeffPred", __FILE__, __LINE__);
	}
      }
	
      // add multimaterial momentum source term

      if (d_MAlab)
	d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					  &velocityVars, &constVelocityVars);

      // Calculate the Velocity BCS
      //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
      //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
      //           [u,v,w]VelNonLinSrcPBLM
      
      if (d_boundaryCondition->anyArchesPhysicalBC()) {
        d_boundaryCondition->velocityBC(pc, patch, 
				        index,
				        cellinfo, &velocityVars,
				        &constVelocityVars);

        if (d_boundaryCondition->getIntrusionBC()) {
	  // if 0'ing stuff below for zero friction drag
#if 0
	  d_boundaryCondition->intrusionMomExchangeBC(pc, patch, index,
						      cellinfo, &velocityVars,
						      &constVelocityVars);
#endif
	  d_boundaryCondition->intrusionVelocityBC(pc, patch, index, 
						   cellinfo, &velocityVars,
						   &constVelocityVars);
	}
      }
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

      if (d_MAlab)
	d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo,
					  &velocityVars, &constVelocityVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      d_source->modifyVelMassSource(pc, patch, delta_t, index,
				    &velocityVars, &constVelocityVars);

      // Calculate Velocity diagonal
      //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM

      d_discretize->calculateVelDiagonal(pc, patch,
					 index,
					 &velocityVars);

      if (d_MAlab) {
	d_boundaryCondition->calculateVelRhoHat_mm(pc, patch, index, delta_t,
						   cellinfo, &velocityVars,
						   &constVelocityVars);
      }
      else {
	d_rhsSolver->calculateHatVelocity(pc, patch, index, delta_t,
					 cellinfo, &velocityVars,
					 &constVelocityVars);
      }

  if (d_pressure_correction) {
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;
  switch(index) {
  case Arches::XDIR:
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;
    fort_computevel(idxLoU, idxHiU, velocityVars.uVelRhoHat, 
		    constVelocityVars.pressure,
		    constVelocityVars.new_density, delta_t,
		    ioff, joff, koff, cellinfo->dxpw);
    break;
  case Arches::YDIR:
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;
    fort_computevel(idxLoU, idxHiU, velocityVars.vVelRhoHat,
		    constVelocityVars.pressure,
		    constVelocityVars.new_density, delta_t,
		    ioff, joff, koff, cellinfo->dyps);

    break;
  case Arches::ZDIR:
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;
    fort_computevel(idxLoU, idxHiU, velocityVars.wVelRhoHat,
		    constVelocityVars.pressure,
		    constVelocityVars.new_density, delta_t,
		    ioff, joff, koff, cellinfo->dzpb);
    break;
  default:
    throw InvalidValue("Invalid index in MomentumSolver::addPressGrad", __FILE__, __LINE__);
  }
  }

    
    }
    double time_shift = 0.0;
    if (d_boundaryCondition->getInletBC()) {
    time_shift = delta_t * timelabels->time_position_multiplier_before_average;
    d_boundaryCondition->velRhoHatInletBC(pc, patch,
					  &velocityVars, &constVelocityVars,
					  time_shift);
    }
    if (d_boundaryCondition->getPressureBC())
    d_boundaryCondition->velRhoHatPressureBC(pc, patch,
					     &velocityVars, &constVelocityVars);
    if (d_boundaryCondition->getOutletBC())
    d_boundaryCondition->velRhoHatOutletBC(pc, patch,
					   &velocityVars, &constVelocityVars);
    /*
  if (d_pressure_correction) {
  int outlet_celltypeval = d_boundaryCondition->outletCellType();
  if (!(outlet_celltypeval==-10)) {
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();


  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constVelocityVars.cellType[xminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
			            constVelocityVars.new_density[xminusCell]);

           velocityVars.uVelRhoHat[currCell] -= 2.0*delta_t*
		   		constVelocityVars.pressure[currCell]/
				(cellinfo->sew[colX] * avdenlow);

           velocityVars.uVelRhoHat[xminusCell] = velocityVars.uVelRhoHat[currCell];

        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constVelocityVars.cellType[xplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[xplusCell] +
			         constVelocityVars.new_density[currCell]);

           velocityVars.uVelRhoHat[xplusCell] += 2.0*delta_t*
				constVelocityVars.pressure[currCell]/
				(cellinfo->sew[colX] * avden);

           velocityVars.uVelRhoHat[xplusplusCell] = velocityVars.uVelRhoHat[xplusCell];
        }
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
        if (constVelocityVars.cellType[yminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
			            constVelocityVars.new_density[yminusCell]);

           velocityVars.vVelRhoHat[currCell] -= 2.0*delta_t*
		   		constVelocityVars.pressure[currCell]/
				(cellinfo->sns[colY] * avdenlow);

           velocityVars.vVelRhoHat[yminusCell] = velocityVars.vVelRhoHat[currCell];

        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constVelocityVars.cellType[yplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[yplusCell] +
			         constVelocityVars.new_density[currCell]);

           velocityVars.vVelRhoHat[yplusCell] += 2.0*delta_t*
		   		constVelocityVars.pressure[currCell]/
				(cellinfo->sns[colY] * avden);

           velocityVars.vVelRhoHat[yplusplusCell] = velocityVars.vVelRhoHat[yplusCell];

        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        if (constVelocityVars.cellType[zminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
			            constVelocityVars.new_density[zminusCell]);

           velocityVars.wVelRhoHat[currCell] -= 2.0*delta_t*
		   		constVelocityVars.pressure[currCell]/
				(cellinfo->stb[colZ] * avdenlow);

           velocityVars.wVelRhoHat[zminusCell] = velocityVars.wVelRhoHat[currCell];

        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constVelocityVars.cellType[zplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[zplusCell] +
			         constVelocityVars.new_density[currCell]);

           velocityVars.wVelRhoHat[zplusCell] += 2.0*delta_t*
		   		constVelocityVars.pressure[currCell]/
				(cellinfo->stb[colZ] * avden);

           velocityVars.wVelRhoHat[zplusplusCell] = velocityVars.wVelRhoHat[zplusCell];

        }
      }
    }
  }
  }
  }*/

//#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch,&velocityVars,&constVelocityVars,
		    d_filter_divergence_constraint,d_3d_periodic);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;
    IntVector ixLow = patch->getCellFORTLowIndex();
    IntVector ixHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
      for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
        for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      velocityVars.divergence[currCell] = (factor_old*old_divergence[currCell]+
			                           factor_new*velocityVars.divergence[currCell])/factor_divide;
        }
      }
    }
//#endif

    
  TAU_PROFILE_STOP(compute);
  }
}

//****************************************************************************
// Schedule the averaging of hat velocities for Runge-Kutta step
//****************************************************************************
void 
MomentumSolver::sched_averageRKHatVelocities(SchedulerP& sched,
					 const PatchSet* patches,
				 	 const MaterialSet* matls,
				         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "MomentumSolver::averageRKHatVelocities" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &MomentumSolver::averageRKHatVelocities,
			  timelabels);

  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually average the Runge-Kutta hat velocities here
//****************************************************************************
void 
MomentumSolver::averageRKHatVelocities(const ProcessorGroup*,
			   	       const PatchSubset* patches,
			   	       const MaterialSubset*,
			   	       DataWarehouse* old_dw,
			   	       DataWarehouse* new_dw,
			   	       const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> old_density;
    constCCVariable<double> temp_density;
    constCCVariable<double> new_density;
    constCCVariable<int> cellType;
    constSFCXVariable<double> old_uvel;
    constSFCYVariable<double> old_vvel;
    constSFCZVariable<double> old_wvel;
    SFCXVariable<double> new_uvel;
    SFCYVariable<double> new_vvel;
    SFCZVariable<double> new_wvel;

    old_dw->get(old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(old_uvel, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(old_vvel, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(old_wvel, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(temp_density, d_lab->d_densityTempLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    new_dw->getModifiable(new_uvel, d_lab->d_uVelRhoHatLabel, 
			  matlIndex, patch);
    new_dw->getModifiable(new_vvel, d_lab->d_vVelRhoHatLabel, 
			  matlIndex, patch);
    new_dw->getModifiable(new_wvel, d_lab->d_wVelRhoHatLabel, 
			  matlIndex, patch);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;

    IntVector indexLow, indexHigh;
    indexLow = patch->getSFCXFORTLowIndex();
    indexHigh = patch->getSFCXFORTHighIndex();

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector xminusCell(colX-1, colY, colZ);
          
	  if (new_density[currCell]<=1.0e-12 || new_density[xminusCell]<=1.0e-12)
	    new_uvel[currCell] = 0.0;
	  else
	    new_uvel[currCell] = (factor_old*old_uvel[currCell]*
		(old_density[currCell]+old_density[xminusCell]) +
		factor_new*new_uvel[currCell]*
		(temp_density[currCell]+temp_density[xminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[xminusCell]));

	}
      }
    }
    indexLow = patch->getSFCYFORTLowIndex();
    indexHigh = patch->getSFCYFORTHighIndex();

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector yminusCell(colX, colY-1, colZ);
          
	  if (new_density[currCell]<=1.0e-12 || new_density[yminusCell]<=1.0e-12)
	    new_vvel[currCell] = 0.0;
	  else
	    new_vvel[currCell] = (factor_old*old_vvel[currCell]*
		(old_density[currCell]+old_density[yminusCell]) +
		factor_new*new_vvel[currCell]*
		(temp_density[currCell]+temp_density[yminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[yminusCell]));

	}
      }
    }
    indexLow = patch->getSFCZFORTLowIndex();
    indexHigh = patch->getSFCZFORTHighIndex();

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector zminusCell(colX, colY, colZ-1);
          
	  if (new_density[currCell]<=1.0e-12 || new_density[zminusCell]<=1.0e-12)
	    new_wvel[currCell] = 0.0;
	  else
	    new_wvel[currCell] = (factor_old*old_wvel[currCell]*
		(old_density[currCell]+old_density[zminusCell]) +
		factor_new*new_wvel[currCell]*
		(temp_density[currCell]+temp_density[zminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[zminusCell]));

	}
      }
    }

// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
  if (d_boundaryCondition->anyArchesPhysicalBC()) {
  int outlet_celltypeval = d_boundaryCondition->outletCellType();
  int pressure_celltypeval = d_boundaryCondition->pressureCellType();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);

 	if (cellType[xminusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    new_uvel[currCell] = 0.0;
	  else {
	  if (old_uvel[currCell] > 1.0e-10)
            new_uvel[currCell] = new_uvel[xplusCell];
	  else
	    new_uvel[currCell] = 0.0;
	  }
	}
	else if (cellType[xminusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    new_uvel[currCell] = 0.0;
	  else {
	  if (old_uvel[currCell] < -1.0e-10)
            new_uvel[currCell] = new_uvel[xplusCell];
	  else
	    new_uvel[currCell] = 0.0;
	  }
	}
	else
	   new_uvel[currCell] = (factor_old*old_uvel[currCell]*
		(old_density[currCell]+old_density[xminusCell]) +
		factor_new*new_uvel[currCell]*
		(temp_density[currCell]+temp_density[xminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[xminusCell]));

 	if ((cellType[xminusCell] == outlet_celltypeval)||
	    (cellType[xminusCell] == pressure_celltypeval)) {
          new_uvel[xminusCell] = new_uvel[currCell];
          /*if (!(yminus && (colY == idxLo.y())))
	    new_vvel[xminusCell] = new_vvel[currCell];
          if (!(zminus && (colZ == idxLo.z())))
	    new_wvel[xminusCell] = new_wvel[currCell];*/
	}
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);

 	if (cellType[xplusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    new_uvel[xplusCell] = 0.0;
	  else {
	  if (old_uvel[xplusCell] < -1.0e-10)
            new_uvel[xplusCell] = new_uvel[currCell];
	  else
	    new_uvel[xplusCell] = 0.0;
	  }
	}
	else if (cellType[xplusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    new_uvel[xplusCell] = 0.0;
	  else {
	  if (old_uvel[xplusCell] > 1.0e-10)
            new_uvel[xplusCell] = new_uvel[currCell];
	  else
	    new_uvel[xplusCell] = 0.0;
	  }
	}
	else
	   new_uvel[xplusCell] = (factor_old*old_uvel[xplusCell]*
		(old_density[xplusCell]+old_density[currCell]) +
		factor_new*new_uvel[xplusCell]*
		(temp_density[xplusCell]+temp_density[currCell]))/
		(factor_divide*(new_density[xplusCell]+new_density[currCell]));

 	if ((cellType[xplusCell] == outlet_celltypeval)||
	    (cellType[xplusCell] == pressure_celltypeval)) {
          new_uvel[xplusplusCell] = new_uvel[xplusCell];
        /*  if (!(yminus && (colY == idxLo.y())))
	    new_vvel[xplusCell] = new_vvel[currCell];
          if (!(zminus && (colZ == idxLo.z())))
	    new_wvel[xplusCell] = new_wvel[currCell];*/
	}
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector yplusCell(colX, colY+1, colZ);

 	if (cellType[yminusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_vvel[currCell] = 0.0;
	  else {
	  if (old_vvel[currCell] > 1.0e-10)
            new_vvel[currCell] = new_vvel[yplusCell];
	  else
	    new_vvel[currCell] = 0.0;
	  }
	}
	else if (cellType[yminusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_vvel[currCell] = 0.0;
	  else {
	  if (old_vvel[currCell] < -1.0e-10)
            new_vvel[currCell] = new_vvel[yplusCell];
	  else
	    new_vvel[currCell] = 0.0;
	  }
	}
	else
	   new_vvel[currCell] = (factor_old*old_vvel[currCell]*
		(old_density[currCell]+old_density[yminusCell]) +
		factor_new*new_vvel[currCell]*
		(temp_density[currCell]+temp_density[yminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[yminusCell]));

 	if ((cellType[yminusCell] == outlet_celltypeval)||
	    (cellType[yminusCell] == pressure_celltypeval)) {
          new_vvel[yminusCell] = new_vvel[currCell];
        /*  if (!(xminus && (colX == idxLo.x())))
	    new_uvel[yminusCell] = new_uvel[currCell];
          if (!(zminus && (colZ == idxLo.z())))
	    new_wvel[yminusCell] = new_wvel[currCell];*/
	}
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);

 	if (cellType[yplusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_vvel[yplusCell] = 0.0;
	  else {
	  if (old_vvel[yplusCell] < -1.0e-10)
            new_vvel[yplusCell] = new_vvel[currCell];
	  else
	    new_vvel[yplusCell] = 0.0;
	  }
	}
	else if (cellType[yplusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_vvel[yplusCell] = 0.0;
	  else {
	  if (old_vvel[yplusCell] > 1.0e-10)
            new_vvel[yplusCell] = new_vvel[currCell];
	  else
	    new_vvel[yplusCell] = 0.0;
	  }
	}
	else
	   new_vvel[yplusCell] = (factor_old*old_vvel[yplusCell]*
		(old_density[yplusCell]+old_density[currCell]) +
		factor_new*new_vvel[yplusCell]*
		(temp_density[yplusCell]+temp_density[currCell]))/
		(factor_divide*(new_density[yplusCell]+new_density[currCell]));

 	if ((cellType[yplusCell] == outlet_celltypeval)||
	    (cellType[yplusCell] == pressure_celltypeval)) {
          new_vvel[yplusplusCell] = new_vvel[yplusCell];
         /* if (!(xminus && (colX == idxLo.x())))
	    new_uvel[yplusCell] = new_uvel[currCell];
          if (!(zminus && (colZ == idxLo.z())))
	    new_wvel[yplusCell] = new_wvel[currCell];*/
	}
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
	IntVector zplusCell(colX, colY, colZ+1);

 	if (cellType[zminusCell] == pressure_celltypeval) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_wvel[currCell] = 0.0;
	  else {
	  if (old_wvel[currCell] > 1.0e-10)
            new_wvel[currCell] = new_wvel[zplusCell];
	  else
	    new_wvel[currCell] = 0.0;
	  }
	}
	else if (cellType[zminusCell] == outlet_celltypeval) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_wvel[currCell] = 0.0;
	  else {
	  if (old_wvel[currCell] < -1.0e-10)
            new_wvel[currCell] = new_wvel[zplusCell];
	  else
	    new_wvel[currCell] = 0.0;
	  }
	}
	else
	   new_wvel[currCell] = (factor_old*old_wvel[currCell]*
		(old_density[currCell]+old_density[zminusCell]) +
		factor_new*new_wvel[currCell]*
		(temp_density[currCell]+temp_density[zminusCell]))/
		(factor_divide*
		(new_density[currCell]+new_density[zminusCell]));

 	if ((cellType[zminusCell] == outlet_celltypeval)||
	    (cellType[zminusCell] == pressure_celltypeval)) {
          new_wvel[zminusCell] = new_wvel[currCell];
        /*  if (!(xminus && (colX == idxLo.x())))
	    new_uvel[zminusCell] = new_uvel[currCell];
          if (!(yminus && (colY == idxLo.y())))
	    new_vvel[zminusCell] = new_vvel[currCell];*/
	}
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);

 	if (cellType[zplusCell] == pressure_celltypeval) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_wvel[zplusCell] = 0.0;
	  else {
	  if (old_wvel[zplusCell] < -1.0e-10)
            new_wvel[zplusCell] = new_wvel[currCell];
	  else
	    new_wvel[zplusCell] = 0.0;
	  }
	}
	else if (cellType[zplusCell] == outlet_celltypeval) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    new_wvel[zplusCell] = 0.0;
	  else {
	  if (old_wvel[zplusCell] > 1.0e-10)
            new_wvel[zplusCell] = new_wvel[currCell];
	  else
	    new_wvel[zplusCell] = 0.0;
	  }
	}
	else
	   new_wvel[zplusCell] = (factor_old*old_wvel[zplusCell]*
		(old_density[zplusCell]+old_density[currCell]) +
		factor_new*new_wvel[zplusCell]*
		(temp_density[zplusCell]+temp_density[currCell]))/
		(factor_divide*(new_density[zplusCell]+new_density[currCell]));

 	if ((cellType[zplusCell] == outlet_celltypeval)||
	    (cellType[zplusCell] == pressure_celltypeval)) {
          new_wvel[zplusplusCell] = new_wvel[zplusCell];
        /*  if (!(xminus && (colX == idxLo.x())))
	    new_uvel[zplusCell] = new_uvel[currCell];
          if (!(yminus && (colY == idxLo.y())))
	    new_vvel[zplusCell] = new_vvel[currCell];*/
	}
      }
    }
  }
  }
  }
}
