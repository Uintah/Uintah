//----- MomentumSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>


using namespace Uintah;
using namespace std;

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
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
MomentumSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
  }

  // make source and boundary_condition objects

  d_source = scinew Source(d_turbModel, d_physicalConsts);

  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs")

    d_linearSolver = scinew RBGSSolver();

  else {

    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);

  }

  d_linearSolver->problemSetup(db);

}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solve(SchedulerP& sched,
		      const PatchSet* patches,
		      const MaterialSet* matls,
		      int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // velocity solve.
  //  DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrix(sched, patches, matls, index);
    
  // Schedules linear velocity solve
  // require : [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM,
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  // compute : [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
  //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
  //           [u,v,w]VelocityMS
  //  d_linearSolver->sched_velSolve(level, sched, new_dw, matrix_dw, index);

  sched_velocityLinearSolve(sched, patches, matls, index);

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
					const MaterialSet* matls,
					int index)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeff",
			  this, &MomentumSolver::buildLinearMatrix,
			  index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // get old_dw from sched
  // from old_dw for time integration

  tsk->requires(Task::OldDW, d_lab->d_refDensity_label);

  // from Task::NewDW

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel,
		Ghost::AroundFaces, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel,
		Ghost::AroundFaces, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel,
		Ghost::AroundFaces, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
		Ghost::AroundCells, numGhostCells);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, numGhostCells);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);

    // for multimaterial

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
			Ghost::None, zeroGhostCells);

    }

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
    }

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
    }

    break;

  default:

    throw InvalidValue("Invalid index in MomentumSolver");
    
  }
	

  /// requires convection coeff because of the nodal
  // differencing
  // computes index components of velocity

  switch (index) {

  case Arches::XDIR:

    tsk->computes(d_lab->d_uVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    tsk->computes(d_lab->d_uVelNonLinSrcMBLMLabel);

    break;

  case Arches::YDIR:

    tsk->computes(d_lab->d_vVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    tsk->computes(d_lab->d_vVelNonLinSrcMBLMLabel);

    break;

  case Arches::ZDIR:

    tsk->computes(d_lab->d_wVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    tsk->computes(d_lab->d_wVelNonLinSrcMBLMLabel);

    break;
    
  default:

    throw InvalidValue("Invalid index in MomentumSolver");

  }

  sched->addTask(tsk, patches, matls);


}

void
MomentumSolver::sched_velocityLinearSolve(SchedulerP& sched, const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("MomentumSolver::VelLinearSolve",
			  this,
			  &MomentumSolver::velocityLinearSolve, index);
  
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  //      DataWarehouseP old_dw = new_dw->getTop();
  // fix for task graph to work

  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);

  switch(index) {

  case Arches::XDIR:

	// for the task graph to work
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel, 
		      Ghost::AroundFaces, numGhostCells+1);
	// coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelNonLinSrcMBLMLabel, 
		  Ghost::None, zeroGhostCells);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
	
    break;

  case Arches::YDIR:

    tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel, 
		  Ghost::AroundFaces, numGhostCells+1);
    // coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelNonLinSrcMBLMLabel, 
		  Ghost::None, zeroGhostCells);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
    break;

  case Arches::ZDIR:

    tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel, 
		  Ghost::AroundFaces, numGhostCells+1);
    // coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelNonLinSrcMBLMLabel, 
		  Ghost::None, zeroGhostCells);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
    break;

  default:

    throw InvalidValue("INValid velocity index for linearSolver");

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
				  int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;

    // Get the required data

    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->getCopy(velocityVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.denRefArray, d_lab->d_denRefArrayLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;

  // get old_dw from getTop function
  //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  // ***checkpoint
  //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();
    sum_vartype den_ref_var;
    old_dw->get(den_ref_var, d_lab->d_refDensity_label);
    velocityVars.den_Ref = den_ref_var;

#ifdef ARCHES_MOM_DEBUG
    cerr << "getdensity_ref in momentum" << velocityVars.den_Ref << endl;
#endif

    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    new_dw->getCopy(velocityVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // for explicit coeffs will be computed using the old u, v, and w
  // change it for implicit solve
  // get void fraction

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, numGhostCells);
    switch (index) {

    case Arches::XDIR:

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.old_uVelocity, d_lab->d_uVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDU, d_lab->d_DUMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocate(velocityVars.uVelocityCoeff[ii], 
			 d_lab->d_uVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(velocityVars.uVelocityConvectCoeff[ii], 
			 d_lab->d_uVelConvCoefMBLMLabel, ii, patch);

      }

      new_dw->allocate(velocityVars.uVelLinearSrc, 
		       d_lab->d_uVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->allocate(velocityVars.uVelNonlinearSrc, 
		       d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocate(velocityVars.pressGradUSu, 
		       d_lab->d_pressGradUSuLabel, matlIndex, patch);

    //cerr << "in moment solve just after allocate" << index << endl;

      break;

    case Arches::YDIR:

      // getting new value of u velocity

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.old_vVelocity, d_lab->d_vVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDV, d_lab->d_DVMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocate(velocityVars.vVelocityCoeff[ii], 
			 d_lab->d_vVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(velocityVars.vVelocityConvectCoeff[ii], 
			 d_lab->d_vVelConvCoefMBLMLabel, ii, patch);

      }

      new_dw->allocate(velocityVars.vVelLinearSrc, 
		       d_lab->d_vVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->allocate(velocityVars.vVelNonlinearSrc, 
		       d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocate(velocityVars.pressGradVSu, 
		       d_lab->d_pressGradVSuLabel, matlIndex, patch);

      break;

    case Arches::ZDIR:

      // getting new value of u velocity

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(velocityVars.old_wVelocity, d_lab->d_wVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

      //    new_dw->allocate(velocityVars.variableCalledDW, d_lab->d_DWMBLMLabel,
      //			matlIndex, patch);
      // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocate(velocityVars.wVelocityCoeff[ii], 
			 d_lab->d_wVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(velocityVars.wVelocityConvectCoeff[ii], 
			 d_lab->d_wVelConvCoefMBLMLabel, ii, patch);

      }

      new_dw->allocate(velocityVars.wVelLinearSrc, 
		       d_lab->d_wVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->allocate(velocityVars.wVelNonlinearSrc, 
		       d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocate(velocityVars.pressGradWSu, 
		       d_lab->d_pressGradWSuLabel, matlIndex, patch);
 
      break;

    default:

      throw InvalidValue("Invalid index in MomentumSolver");

    }

#ifdef ARCHES_MOM_DEBUG
    cerr << "momentum solver: pressure=\n";
    velocityVars.pressure.print(cerr);
    if (patch->containsCell(IntVector(2,3,3))) {
      cerr << "[2,3,3] press[2,3,3]" << velocityVars.pressure[IntVector(2,3,3)] 
	   << " " << velocityVars.pressure[IntVector(1,3,3)] << endl;
    }
#endif
    
    // compute ith componenet of velocity stencil coefficients
    // inputs : [u,v,w]VelocityCPBC, densityIN, viscosityIN
    // outputs: [u,v,w]VelConvCoefPBLM, [u,v,w]VelCoefPBLM

    d_discretize->calculateVelocityCoeff(pc, patch, 
					 delta_t, index,
					 cellinfo, &velocityVars);
    
    // Calculate velocity source
    // inputs : [u,v,w]VelocityCPBC, densityIN, viscosityIN ( new_dw), 
    //          [u,v,w]VelocitySPBC, densityCP( old_dw), 
    // outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->calculateVelocitySource(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars);
    if (d_MAlab)
      d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					&velocityVars);
    
    // Velocity Boundary conditions
    //  inputs : densityIN, [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM

    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &velocityVars);
    
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

    if (d_MAlab)
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, &velocityVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(pc, patch, 
				  delta_t, index,
				  &velocityVars);
    
    // Calculate Velocity Diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelDiagonal(pc, patch, 
				       index,
				       &velocityVars);
    
    // Add the pressure source terms
    // inputs :[u,v,w]VelNonlinSrcMBLM, [u,v,w]VelCoefMBLM, pressureCPBC, 
    //          densityCP(old_dw), 
    // [u,v,w]VelocityCPBC
    // outputs:[u,v,w]VelNonlinSrcMBLM

    // for explicit this is not requires
    //  d_source->addTransMomSource(pc, patch, delta_t, index, cellinfo,
    //			      &velocityVars);
    d_source->computePressureSource(pc, patch, index,
				    cellinfo, &velocityVars);
    d_boundaryCondition->addPressureGrad(pc, patch, index, cellinfo,
					 &velocityVars);

#ifdef ARCHES_MOM_DEBUG
    if (index == 1) {
      cerr << "After vel voef for u" << endl;
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uAE"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AE][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uAW"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AW][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uAN"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AN][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uAS"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AS][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
      cerr.width(10);
      cerr << "uAT"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AT][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
      cerr.width(10);
      cerr << "uAB"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AB][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uAP"<<*iter << ": " << velocityVars.uVelocityCoeff[Arches::AP][*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uSU"<<*iter << ": " << velocityVars.uVelNonlinearSrc[*iter] << "\n" ; 
      }
      for(CellIterator iter = patch->getCellIterator();
	  !iter.done(); iter++){
	cerr.width(10);
	cerr << "uSP"<<*iter << ": " << velocityVars.uVelLinearSrc[*iter] << "\n" ; 
      }
    }
#endif
    
    //cerr << "in moment solve just before build matrix" << index << endl;
    // put required vars

    switch (index) {

    case Arches::XDIR:

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
	new_dw->put(velocityVars.uVelocityCoeff[ii], 
		    d_lab->d_uVelCoefMBLMLabel, ii, patch);
	
      }
      new_dw->put(velocityVars.uVelNonlinearSrc, 
		  d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;

    case Arches::YDIR:

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
	new_dw->put(velocityVars.vVelocityCoeff[ii], 
		    d_lab->d_vVelCoefMBLMLabel, ii, patch);
    }
      new_dw->put(velocityVars.vVelNonlinearSrc, 
		  d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
      
    case Arches::ZDIR:

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
	new_dw->put(velocityVars.wVelocityCoeff[ii], 
		     d_lab->d_wVelCoefMBLMLabel, ii, patch);
      }
      new_dw->put(velocityVars.wVelNonlinearSrc, 
		  d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;

    default:

      throw InvalidValue("Invalid index in MomentumSolver");

    }
  }
}

void 
MomentumSolver::velocityLinearSolve(const ProcessorGroup* pc,
				    const PatchSubset* patches,
				    const MaterialSubset* /*matls*/,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw,
				    int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    //  DataWarehouseP old_dw = new_dw->getTop();
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // get old_dw from getTop function
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    // old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    switch (index) {
    case Arches::XDIR:
      {
      new_dw->allocate(velocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->getCopy(velocityVars.uVelocityCoeff[ii], 
		    d_lab->d_uVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, zeroGhostCells);
      new_dw->getCopy(velocityVars.uVelNonlinearSrc, 
		  d_lab->d_uVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, zeroGhostCells);
      new_dw->allocate(velocityVars.residualUVelocity, d_lab->d_uVelocityRes,
		       matlIndex, patch);
      
      break;
    case Arches::YDIR:
      {
      new_dw->allocate(velocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel,
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }
      // initial guess for explicit calculations
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->getCopy(velocityVars.vVelocityCoeff[ii], 
		    d_lab->d_vVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, zeroGhostCells);
      new_dw->getCopy(velocityVars.vVelNonlinearSrc, 
		  d_lab->d_vVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, zeroGhostCells);
      new_dw->allocate(velocityVars.residualVVelocity, d_lab->d_vVelocityRes,
		       matlIndex, patch);
      break; 
    case Arches::ZDIR:
      {
      new_dw->allocate(velocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->getCopy(velocityVars.wVelocityCoeff[ii], 
		    d_lab->d_wVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, zeroGhostCells);
      new_dw->getCopy(velocityVars.wVelNonlinearSrc, 
		  d_lab->d_wVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, zeroGhostCells);
      new_dw->allocate(velocityVars.residualWVelocity, d_lab->d_wVelocityRes,
		       matlIndex, patch);
      break;  
    default:
      throw InvalidValue("Invalid index in MomentumSolver");
    }
  
  
  // compute eqn residual
#if 0
    d_linearSolver->computeVelResidual(pc, patch, new_dw, new_dw, index, 
				     &velocityVars);
    // put the summed residuals into new_dw
    switch (index) {
    case Arches::XDIR:
      new_dw->put(sum_vartype(velocityVars.residUVel), d_lab->d_uVelResidPSLabel);
      new_dw->put(sum_vartype(velocityVars.truncUVel), d_lab->d_uVelTruncPSLabel);
      break;
    case Arches::YDIR:
      new_dw->put(sum_vartype(velocityVars.residVVel), d_lab->d_vVelResidPSLabel);
      new_dw->put(sum_vartype(velocityVars.truncVVel), d_lab->d_vVelTruncPSLabel);
      break;
    case Arches::ZDIR:
      new_dw->put(sum_vartype(velocityVars.residWVel), d_lab->d_wVelResidPSLabel);
      new_dw->put(sum_vartype(velocityVars.truncWVel), d_lab->d_wVelTruncPSLabel);
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");  
    }
#endif
    
    // apply underelax to eqn
    d_linearSolver->computeVelUnderrelax(pc, patch,  index, 
					 &velocityVars);
    // old_velocities for explicit calculation
    
    switch(index){
    case Arches::XDIR:
      velocityVars.old_uVelocity.allocate(velocityVars.uVelocity.getLowIndex(),
					  velocityVars.uVelocity.getHighIndex());
      velocityVars.old_uVelocity.copy(velocityVars.uVelocity);
      break;
    case Arches::YDIR:
      velocityVars.old_vVelocity.allocate(velocityVars.vVelocity.getLowIndex(),
					  velocityVars.vVelocity.getHighIndex());
      velocityVars.old_vVelocity.copy(velocityVars.vVelocity);
      break;
    case Arches::ZDIR:
      velocityVars.old_wVelocity.allocate(velocityVars.wVelocity.getLowIndex(),
					  velocityVars.wVelocity.getHighIndex());
      velocityVars.old_wVelocity.copy(velocityVars.wVelocity);
      break;
    }
    
    
    // make it a separate task later
    d_linearSolver->velocityLisolve(pc, patch, index, delta_t, 
				    &velocityVars, cellinfo, d_lab);
#if 0
    cerr << "Print computed velocity: " <<index<< endl;
    if (index == 1)
      velocityVars.uVelocity.print(cerr);
    else if (index == 2)
      velocityVars.vVelocity.print(cerr);
#endif

    // put back the results
    switch (index) {
    case Arches::XDIR:
      new_dw->put(velocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch);
      break;
    case Arches::YDIR:
      new_dw->put(velocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel,
		  matlIndex, patch);
      break;
    case Arches::ZDIR:
      new_dw->put(velocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch);
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");  
    }
  }
}


// predictor step
//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solvePred(SchedulerP& sched,
			  const PatchSet* patches,
			  const MaterialSet* matls,
			  int index)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrixPred(sched, patches, matls, index);
    

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixPred(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls,
					    int index)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeffPred",
			  this, &MomentumSolver::buildLinearMatrixPred,
			  index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
#ifdef correctorstep
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel,
		Ghost::AroundCells, numGhostCells);
#else
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, numGhostCells);

  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, numGhostCells);
#endif
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, numGhostCells);
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, numGhostCells);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		  Ghost::AroundFaces, numGhostCells);
#ifdef correctorstep
    tsk->computes(d_lab->d_uVelocityPredLabel);
#else
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
#endif

    // for multimaterial

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
			Ghost::None, zeroGhostCells);

    }

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel, 
		  Ghost::AroundFaces, numGhostCells);
#ifdef correctorstep
    tsk->computes(d_lab->d_vVelocityPredLabel);
#else
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
#endif
    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
    }

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel, 
		  Ghost::AroundFaces, numGhostCells);
#ifdef correctorstep
    tsk->computes(d_lab->d_wVelocityPredLabel);
#else
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
#endif
    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
    }

    break;

  default:

    throw InvalidValue("Invalid index in MomentumSolver");
    
  }
	
  sched->addTask(tsk, patches, matls);


}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void 
MomentumSolver::buildLinearMatrixPred(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
#ifdef correctorstep
  delta_t /= 2.0;
#endif
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;

    // Get the required data
#ifdef correctorstep
    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

#else
    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

#endif

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, numGhostCells);
    switch (index) {

    case Arches::XDIR:

#ifdef correctorstep
      new_dw->allocate(velocityVars.uVelRhoHat, d_lab->d_uVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#else
      new_dw->allocate(velocityVars.uVelRhoHat, d_lab->d_uVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#endif
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }

      break;

    case Arches::YDIR:

      // getting new value of u velocity
#ifdef correctorstep
      new_dw->allocate(velocityVars.vVelRhoHat, d_lab->d_vVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#else
      new_dw->allocate(velocityVars.vVelRhoHat, d_lab->d_vVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#endif
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }
      
      break;

    case Arches::ZDIR:

      // getting new value of u velocity
#ifdef correctorstep
      new_dw->allocate(velocityVars.wVelRhoHat, d_lab->d_wVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#else
      new_dw->allocate(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#endif
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->getCopy(velocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);

      }

      break;

    default:

      throw InvalidValue("Invalid index in MomentumSolver");

    }

#ifdef ARCHES_MOM_DEBUG
    cerr << "momentum solver: pressure=\n";
    velocityVars.pressure.print(cerr);
    if (patch->containsCell(IntVector(2,3,3))) {
      cerr << "[2,3,3] press[2,3,3]" << velocityVars.pressure[IntVector(2,3,3)] 
	   << " " << velocityVars.pressure[IntVector(1,3,3)] << endl;
    }
#endif
    
    d_source->calculateVelocityPred(pc, patch, 
				    delta_t, index,
				    cellinfo, &velocityVars);
    switch (index) {
    case Arches::XDIR:
#if 0
      cerr << "Print uvelRhoHat after solve" << endl;
      velocityVars.uVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocityPredLabel, matlIndex, patch);
#else
      new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
#endif
      break;
    case Arches::YDIR:
#if 0
      cerr << "Print vvelRhoHat after solve" << endl;
      velocityVars.vVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocityPredLabel, matlIndex, patch);
#else
      new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocitySPBCLabel, matlIndex, patch);

#endif
      break;
    case Arches::ZDIR:
#if 0
      cerr << "Print wvelRhoHat after solve" << endl;
      velocityVars.wVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocityPredLabel, matlIndex, patch);
#else
      new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
#endif
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");
    }

  }
}

// Corrector step
//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solveCorr(SchedulerP& sched,
			  const PatchSet* patches,
			  const MaterialSet* matls,
			  int index)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrixCorr(sched, patches, matls, index);
    

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixCorr(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls,
					    int index)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeffCorr",
			  this, &MomentumSolver::buildLinearMatrixCorr,
			  index);
  int numGhostCells = 1;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, numGhostCells);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, numGhostCells);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatCorrLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, numGhostCells);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);

    break;

  default:

    throw InvalidValue("Invalid index in MomentumSolver");
    
  }
	
  sched->addTask(tsk, patches, matls);


}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void 
MomentumSolver::buildLinearMatrixCorr(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    int numGhostCells = 1;

    // Get the required data

    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, numGhostCells);
    switch (index) {

    case Arches::XDIR:

      {
      new_dw->allocate(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }

      break;

    case Arches::YDIR:

      // getting new value of u velocity

      {
      new_dw->allocate(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }

      break;

    case Arches::ZDIR:

      // getting new value of u velocity

      {
      new_dw->allocate(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, numGhostCells);
      }
      break;

    default:

      throw InvalidValue("Invalid index in MomentumSolver");

    }

#ifdef ARCHES_MOM_DEBUG
    cerr << "momentum solver: pressure=\n";
    velocityVars.pressure.print(cerr);
    if (patch->containsCell(IntVector(2,3,3))) {
      cerr << "[2,3,3] press[2,3,3]" << velocityVars.pressure[IntVector(2,3,3)] 
	   << " " << velocityVars.pressure[IntVector(1,3,3)] << endl;
    }
#endif
    
    d_source->calculateVelocityPred(pc, patch, 
				    delta_t, index,
				    cellinfo, &velocityVars);
    switch (index) {
    case Arches::XDIR:
#if 0
      cerr << "Print uvelRhoHat after solve" << endl;
      velocityVars.uVelRhoHat.print(cerr);
#endif
      new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
      break;
    case Arches::YDIR:
#if 0
      cerr << "Print vvelRhoHat after solve" << endl;
      velocityVars.vVelRhoHat.print(cerr);
#endif
      new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
      break;
    case Arches::ZDIR:
#if 0
      cerr << "Print wvelRhoHat after solve" << endl;
      velocityVars.wVelRhoHat.print(cerr);
#endif
      new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");
    }

  }
}

