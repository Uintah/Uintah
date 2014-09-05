//----- MomentumSolver.cc ----------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
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
#include <Core/Math/MiscMath.h>


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
  db->getWithDefault("central",d_central,false);
//  if (db->findBlock("central"))
//    db->require("central",d_central);
//  else
//    d_central = false;
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

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // get old_dw from sched
  // from old_dw for time integration

  tsk->requires(Task::OldDW, d_lab->d_refDensity_label);

  // from Task::NewDW

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // for multimaterial

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
			Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
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
  

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  //      DataWarehouseP old_dw = new_dw->getTop();
  // fix for task graph to work

  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch(index) {

  case Arches::XDIR:

	// for the task graph to work
//    tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel, 
//		      Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	// coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_uVelNonLinSrcMBLMLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
	
    break;

  case Arches::YDIR:

//    tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel, 
//		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    // coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelNonLinSrcMBLMLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
    break;

  case Arches::ZDIR:

//    tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel, 
//		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    // coefficient for the variable for which solve is invoked
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelCoefMBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelNonLinSrcMBLMLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
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
    ArchesConstVariables constVelocityVars;

    // Get the required data

    new_dw->get(constVelocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayLabel,
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

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

    new_dw->get(constVelocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // for explicit coeffs will be computed using the old u, v, and w
  // change it for implicit solve
  // get void fraction

    if (d_MAlab)
      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    switch (index) {

    case Arches::XDIR:

      new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDU, d_lab->d_DUMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->get(constVelocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->get(constVelocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

      }

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocateAndPut(velocityVars.uVelocityCoeff[ii], d_lab->d_uVelCoefMBLMLabel, ii, patch);
	new_dw->allocateTemporary(velocityVars.uVelocityConvectCoeff[ii],  patch);

      }

      new_dw->allocateTemporary(velocityVars.uVelLinearSrc,  patch);
      new_dw->allocateAndPut(velocityVars.uVelNonlinearSrc, d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocateTemporary(velocityVars.pressGradUSu,  patch);

    //cerr << "in moment solve just after allocate" << index << endl;

      break;

    case Arches::YDIR:

      // getting new value of u velocity

      new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDV, d_lab->d_DVMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->get(constVelocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->get(constVelocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocateAndPut(velocityVars.vVelocityCoeff[ii], d_lab->d_vVelCoefMBLMLabel, ii, patch);
	new_dw->allocateTemporary(velocityVars.vVelocityConvectCoeff[ii],  patch);

      }

      new_dw->allocateTemporary(velocityVars.vVelLinearSrc,  patch);
      new_dw->allocateAndPut(velocityVars.vVelNonlinearSrc, d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocateTemporary(velocityVars.pressGradVSu,  patch);

      break;

    case Arches::ZDIR:

      // getting new value of u velocity

      new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      //    new_dw->allocate(velocityVars.variableCalledDW, d_lab->d_DWMBLMLabel,
      //			matlIndex, patch);
      // for multimaterial

      if (d_MAlab) {

	new_dw->get(constVelocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->get(constVelocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

      }

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	new_dw->allocateAndPut(velocityVars.wVelocityCoeff[ii], d_lab->d_wVelCoefMBLMLabel, ii, patch);
	new_dw->allocateTemporary(velocityVars.wVelocityConvectCoeff[ii],  patch);

      }

      new_dw->allocateTemporary(velocityVars.wVelLinearSrc,  patch);
      new_dw->allocateAndPut(velocityVars.wVelNonlinearSrc, d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch);

      // for computing pressure gradient for momentum source

      new_dw->allocateTemporary(velocityVars.pressGradWSu,  patch);
 
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
					 delta_t, index, d_central,
					 cellinfo, &velocityVars,
					 &constVelocityVars);
    
    // Calculate velocity source
    // inputs : [u,v,w]VelocityCPBC, densityIN, viscosityIN ( new_dw), 
    //          [u,v,w]VelocitySPBC, densityCP( old_dw), 
    // outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->calculateVelocitySource(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars,
				      &constVelocityVars);
    if (d_MAlab)
      d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					&velocityVars, &constVelocityVars);
    
    // Velocity Boundary conditions
    //  inputs : densityIN, [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM

    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &velocityVars,
				    &constVelocityVars);
    
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

    if (d_MAlab)
	d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo,
					  &velocityVars, &constVelocityVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(pc, patch, 
				  delta_t, index,
				  &velocityVars,
				  &constVelocityVars);
    
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
    // apply underelax to eqn
    d_linearSolver->computeVelUnderrelax(pc, patch,  index, 
					 &velocityVars, &constVelocityVars);

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
	// allocateAndPut instead:
	/* new_dw->put(velocityVars.uVelocityCoeff[ii], 
		    d_lab->d_uVelCoefMBLMLabel, ii, patch); */;
	
      }
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelNonlinearSrc, 
		  d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch); */;
      break;

    case Arches::YDIR:

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
	// allocateAndPut instead:
	/* new_dw->put(velocityVars.vVelocityCoeff[ii], 
		    d_lab->d_vVelCoefMBLMLabel, ii, patch); */;
    }
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelNonlinearSrc, 
		  d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch); */;
      break;
      
    case Arches::ZDIR:

      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
	// allocateAndPut instead:
	/* new_dw->put(velocityVars.wVelocityCoeff[ii], 
		     d_lab->d_wVelCoefMBLMLabel, ii, patch); */;
      }
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelNonlinearSrc, 
		  d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch); */;
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
    ArchesConstVariables constVelocityVars;
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
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    switch (index) {
    case Arches::XDIR:
      {
      new_dw->allocateAndPut(velocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->get(constVelocityVars.uVelocityCoeff[ii], 
		    d_lab->d_uVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(constVelocityVars.uVelNonlinearSrc, 
		  d_lab->d_uVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateTemporary(velocityVars.residualUVelocity,  patch);
      
      break;
    case Arches::YDIR:
      {
      new_dw->allocateAndPut(velocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel,
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }
      // initial guess for explicit calculations
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->get(constVelocityVars.vVelocityCoeff[ii], 
		    d_lab->d_vVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(constVelocityVars.vVelNonlinearSrc, 
		  d_lab->d_vVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateTemporary(velocityVars.residualVVelocity,  patch);
      break; 
    case Arches::ZDIR:
      {
      new_dw->allocateAndPut(velocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
	new_dw->get(constVelocityVars.wVelocityCoeff[ii], 
		    d_lab->d_wVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(constVelocityVars.wVelNonlinearSrc, 
		  d_lab->d_wVelNonLinSrcMBLMLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateTemporary(velocityVars.residualWVelocity,  patch);
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
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch); */;
      break;
    case Arches::YDIR:
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel,
		  matlIndex, patch); */;
      break;
    case Arches::ZDIR:
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch); */;
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

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
#ifdef correctorstep
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  if (d_MAlab) {

    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  }

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef correctorstep
    tsk->computes(d_lab->d_uVelocityPredLabel);
#else
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
#endif

// This should only be done when ENO is used for scalar
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsUPred_label);
#else
    tsk->computes(d_lab->d_maxAbsU_label);
#endif

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef correctorstep
    tsk->computes(d_lab->d_vVelocityPredLabel);
#else
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
#endif

// This should only be done when ENO is used for scalar
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsVPred_label);
#else
    tsk->computes(d_lab->d_maxAbsV_label);
#endif

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef correctorstep
    tsk->computes(d_lab->d_wVelocityPredLabel);
#else
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
#endif

// This should only be done when ENO is used for scalar
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsWPred_label);
#else
    tsk->computes(d_lab->d_maxAbsW_label);
#endif

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
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
#endif

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // Get the required data
#ifdef correctorstep
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#else
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

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

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,
		    matlIndex, patch, 
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (d_MAlab) {

      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		      matlIndex, patch, 
		      Ghost::AroundCells, Arches::ONEGHOSTCELL);

    }
    
    switch (index) {

    case Arches::XDIR:

#ifdef correctorstep
      new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#else
      new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#endif
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

      break;

    case Arches::YDIR:

      // getting new value of u velocity
#ifdef correctorstep
      new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#else
      new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#endif
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // for multimaterial

      break;

    case Arches::ZDIR:

      // getting new value of u velocity
#ifdef correctorstep
      new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelocityPredLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#else
      new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel,
		       matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#endif
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      // for multimaterial

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
    
    // Actual compute operations

    if (d_MAlab) {
      
      d_boundaryCondition->calculateVelocityPred_mm(pc, patch, 
						    delta_t, index, cellinfo,
						    &velocityVars,
						    &constVelocityVars);

    }
    else {
    
      d_source->calculateVelocityPred(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars,
				      &constVelocityVars);

      if (d_boundaryCondition->getIntrusionBC())
	d_boundaryCondition->calculateIntrusionVel(pc, patch,
						   index, cellinfo,
						   &velocityVars,
						   &constVelocityVars);
    }
    d_boundaryCondition->velocityPressureBC(pc, patch, index, cellinfo,
					    &velocityVars, &constVelocityVars);
    d_boundaryCondition->addPresGradVelocityOutletBC(pc, patch, index, cellinfo,
						     delta_t, &velocityVars,
						     &constVelocityVars);

// This should only be done when ENO is used for scalar
    double maxAbsU = 0.0;
    double maxAbsV = 0.0;
    double maxAbsW = 0.0;
    double temp_absU, temp_absV, temp_absW;
    IntVector ixLow;
    IntVector ixHigh;
    
    switch (index) {
    case Arches::XDIR:

      ixLow = patch->getSFCXFORTLowIndex();
      ixHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absU = Abs(velocityVars.uVelRhoHat[currCell]);
	      if (temp_absU > maxAbsU) maxAbsU = temp_absU;
          }
        }
      }
      #ifdef correctorstep
      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsUPred_label); 
      #else
      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsU_label); 
      #endif

      break;
    case Arches::YDIR:

      ixLow = patch->getSFCYFORTLowIndex();
      ixHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absV = Abs(velocityVars.vVelRhoHat[currCell]);
	      if (temp_absV > maxAbsV) maxAbsV = temp_absV;
          }
        }
      }
      #ifdef correctorstep
      new_dw->put(max_vartype(maxAbsV), d_lab->d_maxAbsVPred_label); 
      #else
      new_dw->put(max_vartype(maxAbsV), d_lab->d_maxAbsV_label); 
      #endif

      break;
    case Arches::ZDIR:

      ixLow = patch->getSFCZFORTLowIndex();
      ixHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absW = Abs(velocityVars.wVelRhoHat[currCell]);
	      if (temp_absW > maxAbsW) maxAbsW = temp_absW;
          }
        }
      }
      #ifdef correctorstep
      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsWPred_label); 
      #else
      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsW_label); 
      #endif

      break;
    default:
      throw InvalidValue("Invalid index in max abs velocity calculation");
    }

    switch (index) {
    case Arches::XDIR:
#if 0
      cerr << "Print uvelRhoHat after solve" << endl;
      velocityVars.uVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocityPredLabel, matlIndex, patch); */;
#else
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocitySPBCLabel, matlIndex, patch); */;
#endif
      break;
    case Arches::YDIR:
#if 0
      cerr << "Print vvelRhoHat after solve" << endl;
      velocityVars.vVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocityPredLabel, matlIndex, patch); */;
#else
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocitySPBCLabel, matlIndex, patch); */;

#endif
      break;
    case Arches::ZDIR:
#if 0
      cerr << "Print wvelRhoHat after solve" << endl;
      velocityVars.wVelRhoHat.print(cerr);
#endif
#ifdef correctorstep
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocityPredLabel, matlIndex, patch); */;
#else
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocitySPBCLabel, matlIndex, patch); */;
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

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatCorrLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsU_label);

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsV_label);

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsW_label);

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

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  double zeta_2 = -5.0/12.0;
  delta_t *= (gamma_3+zeta_2);
#endif
#endif

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // Get the required data

    new_dw->get(constVelocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityCPLabel, 
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

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,
		      matlIndex, patch, 
		      Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (d_MAlab)
      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    switch (index) {

    case Arches::XDIR:

      {
      new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }

      break;

    case Arches::YDIR:

      // getting new value of u velocity

      {
      new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }

      break;

    case Arches::ZDIR:

      // getting new value of u velocity

      {
      new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
				    cellinfo, &velocityVars,
				    &constVelocityVars);

    d_boundaryCondition->velocityPressureBC(pc, patch, index, cellinfo,
					    &velocityVars, &constVelocityVars);
    d_boundaryCondition->addPresGradVelocityOutletBC(pc, patch, index, cellinfo,
						     delta_t, &velocityVars,
						     &constVelocityVars);

// This should only be done when ENO is used for scalar
    double maxAbsU = 0.0;
    double maxAbsV = 0.0;
    double maxAbsW = 0.0;
    double temp_absU, temp_absV, temp_absW;
    IntVector ixLow;
    IntVector ixHigh;
    
    switch (index) {
    case Arches::XDIR:

      ixLow = patch->getSFCXFORTLowIndex();
      ixHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absU = Abs(velocityVars.uVelRhoHat[currCell]);
	      if (temp_absU > maxAbsU) maxAbsU = temp_absU;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsU_label); 

      break;
    case Arches::YDIR:

      ixLow = patch->getSFCYFORTLowIndex();
      ixHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absV = Abs(velocityVars.vVelRhoHat[currCell]);
	      if (temp_absV > maxAbsV) maxAbsV = temp_absV;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsV), d_lab->d_maxAbsV_label); 

      break;
    case Arches::ZDIR:

      ixLow = patch->getSFCZFORTLowIndex();
      ixHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absW = Abs(velocityVars.wVelRhoHat[currCell]);
	      if (temp_absW > maxAbsW) maxAbsW = temp_absW;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsW_label); 

      break;
    default:
      throw InvalidValue("Invalid index in max abs velocity calculation");
    }

    switch (index) {
    case Arches::XDIR:
#if 0
      cerr << "Print uvelRhoHat after solve" << endl;
      velocityVars.uVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocitySPBCLabel, matlIndex, patch); */;
      break;
    case Arches::YDIR:
#if 0
      cerr << "Print vvelRhoHat after solve" << endl;
      velocityVars.vVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocitySPBCLabel, matlIndex, patch); */;
      break;
    case Arches::ZDIR:
#if 0
      cerr << "Print wvelRhoHat after solve" << endl;
      velocityVars.wVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocitySPBCLabel, matlIndex, patch); */;
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");
    }

  }
}

//****************************************************************************
// Schedule linear momentum solve, intermediate step
//****************************************************************************
void 
MomentumSolver::solveInterm(SchedulerP& sched,
			  const PatchSet* patches,
			  const MaterialSet* matls,
			  int index)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrixInterm(sched, patches, matls, index);
    

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixInterm(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls,
					    int index)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeffInterm",
			  this, &MomentumSolver::buildLinearMatrixInterm,
			  index);

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_uVelocityIntermLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsUInterm_label);

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_vVelocityIntermLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsVInterm_label);

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->computes(d_lab->d_wVelocityIntermLabel);

// This should only be done when ENO is used for scalar
    tsk->computes(d_lab->d_maxAbsWInterm_label);

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
MomentumSolver::buildLinearMatrixInterm(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

  #ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  double zeta_1 = -17.0/60.0;
  delta_t *= (gamma_2+zeta_1); 
  #endif

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // Get the required data

    new_dw->get(constVelocityVars.pressure, d_lab->d_pressureIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityIntermLabel, 
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

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,
		      matlIndex, patch, 
		      Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (d_MAlab)
      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    switch (index) {

    case Arches::XDIR:

      {
      new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelocityIntermLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }

      break;

    case Arches::YDIR:

      // getting new value of u velocity

      {
      new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelocityIntermLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      }

      break;

    case Arches::ZDIR:

      // getting new value of u velocity

      {
      new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelocityIntermLabel, matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
				    cellinfo, &velocityVars,
				    &constVelocityVars);

    d_boundaryCondition->velocityPressureBC(pc, patch, index, cellinfo,
					    &velocityVars, &constVelocityVars);
    d_boundaryCondition->addPresGradVelocityOutletBC(pc, patch, index, cellinfo,
						     delta_t, &velocityVars,
						     &constVelocityVars);

// This should only be done when ENO is used for scalar
    double maxAbsU = 0.0;
    double maxAbsV = 0.0;
    double maxAbsW = 0.0;
    double temp_absU, temp_absV, temp_absW;
    IntVector ixLow;
    IntVector ixHigh;
    
    switch (index) {
    case Arches::XDIR:

      ixLow = patch->getSFCXFORTLowIndex();
      ixHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absU = Abs(velocityVars.uVelRhoHat[currCell]);
	      if (temp_absU > maxAbsU) maxAbsU = temp_absU;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsUInterm_label); 

      break;
    case Arches::YDIR:

      ixLow = patch->getSFCYFORTLowIndex();
      ixHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absV = Abs(velocityVars.vVelRhoHat[currCell]);
	      if (temp_absV > maxAbsV) maxAbsV = temp_absV;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsV), d_lab->d_maxAbsVInterm_label); 

      break;
    case Arches::ZDIR:

      ixLow = patch->getSFCZFORTLowIndex();
      ixHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = ixLow.z(); colZ <= ixHigh.z(); colZ ++) {
        for (int colY = ixLow.y(); colY <= ixHigh.y(); colY ++) {
          for (int colX = ixLow.x(); colX <= ixHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absW = Abs(velocityVars.wVelRhoHat[currCell]);
	      if (temp_absW > maxAbsW) maxAbsW = temp_absW;
          }
        }
      }

      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsWInterm_label); 

      break;
    default:
      throw InvalidValue("Invalid index in max abs velocity calculation");
    }

    switch (index) {
    case Arches::XDIR:
#if 0
      cerr << "Print uvelRhoHat after solve" << endl;
      velocityVars.uVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.uVelRhoHat, 
		  d_lab->d_uVelocityIntermLabel, matlIndex, patch); */;
      break;
    case Arches::YDIR:
#if 0
      cerr << "Print vvelRhoHat after solve" << endl;
      velocityVars.vVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.vVelRhoHat, 
		  d_lab->d_vVelocityIntermLabel, matlIndex, patch); */;
      break;
    case Arches::ZDIR:
#if 0
      cerr << "Print wvelRhoHat after solve" << endl;
      velocityVars.wVelRhoHat.print(cerr);
#endif
      // allocateAndPut instead:
      /* new_dw->put(velocityVars.wVelRhoHat,
		  d_lab->d_wVelocityIntermLabel, matlIndex, patch); */;
      break;
    default:
      throw InvalidValue("Invalid index in MomentumSolver");
    }

  }
}


void MomentumSolver::solveVelHatPred(const LevelP& level,
				     SchedulerP& sched,
				     const int Runge_Kutta_current_step,
				     const bool Runge_Kutta_last_step)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

#ifdef filter_convection_terms
  sched_computeNonlinearTerms(sched, patches, matls, d_lab, 
		Runge_Kutta_current_step, Runge_Kutta_last_step);
#endif

  sched_buildLinearMatrixVelHatPred(sched, patches, matls);

}



// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixVelHatPred(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls)
{
  Task* tsk = scinew Task( "Momentumsolve::BuildCoeffVelHatPred", 
			   this, &MomentumSolver::buildLinearMatrixVelHatPred);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }

#ifdef do_pressure_correction
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
    tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel,
		  d_lab->d_stressTensorMatl,Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef correctorstep
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  }
#if 0
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  // required for computing div constraint
#ifdef divergenceconstraint
#ifdef correctorstep
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_drhodfPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#else
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefSrcPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

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

#ifdef filter_convection_terms
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjULabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjVLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjWLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
#endif
    
    
  // requires convection coeff because of the nodal
  // differencing
  // computes all the components of velocity

  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);
#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  tsk->computes(d_lab->d_uVelTempLabel);
  tsk->computes(d_lab->d_vVelTempLabel);
  tsk->computes(d_lab->d_wVelTempLabel);
#endif
#endif
  //  tsk->computes(d_lab->d_velocityDivergenceLabel);
  //  tsk->computes(d_lab->d_velocityDivergenceBCLabel);
    
#ifdef divergenceconstraint
  tsk->computes(d_lab->d_divConstraintLabel);
#endif
  sched->addTask(tsk, patches, matls);
}




// ***********************************************************************
// Actual build of linear matrices for momentum components
// ***********************************************************************

void 
MomentumSolver::buildLinearMatrixVelHatPred(const ProcessorGroup* pc,
					    const PatchSubset* patches,
					    const MaterialSubset* /*matls*/,
					    DataWarehouse* old_dw,
					    DataWarehouse* new_dw)
{
  TAU_PROFILE_TIMER(input, "Input", "[MomSolver::buildMVelHatPred::input]" , TAU_USER);
  TAU_PROFILE_TIMER(inputcell, "Inputcell", "[MomSolver::buildMVelHatPred::inputcell]" , TAU_USER);
  TAU_PROFILE_TIMER(compute, "Compute", "[MomSolver::buildMVelHatPred::compute]" , TAU_USER);
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifdef correctorstep
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {
  TAU_PROFILE_START(input);

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // compute all three componenets of velocity stencil coefficients


    // Get the reference density
    // Get the required data

    new_dw->get(constVelocityVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef correctorstep
    new_dw->get(constVelocityVars.new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayPredLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->get(constVelocityVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef do_pressure_correction
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif

#ifdef divergenceconstraint
#ifdef correctorstep
    new_dw->get(constVelocityVars.scalar, d_lab->d_scalarPredLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.drhodf, d_lab->d_drhodfPredLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#else
    new_dw->get(constVelocityVars.scalar, d_lab->d_scalarOUTBCLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.drhodf, d_lab->d_drhodfCPLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constVelocityVars.scalarDiffusionCoeff[ii],
		      d_lab->d_scalDiffCoefPredLabel,
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.scalarDiffNonlinearSrc, 
	        d_lab->d_scalDiffCoefSrcPredLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(velocityVars.divergence, d_lab->d_divConstraintLabel,
		     matlIndex, patch);
    velocityVars.divergence.initialize(0.0);
#endif

  TAU_PROFILE_STOP(input);
  TAU_PROFILE_START(inputcell);
    PerPatch<CellInformationP> cellInfoP;

    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();
  TAU_PROFILE_STOP(inputcell);
  TAU_PROFILE_START(input);

    if (d_MAlab) {
      new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
      new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
      new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    }
    else {
      new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    }
    new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (d_MAlab) {
      new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		      matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    }
    else {
      new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
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
	  throw InvalidValue("invalid index for velocity in MomentumSolver");
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
	new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.uVelRhoHat.copy(constVelocityVars.uVelocity,
				     velocityVars.uVelRhoHat.getLowIndex(),
				     velocityVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(velocityVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.vVelRhoHat.copy(constVelocityVars.vVelocity,
				     velocityVars.vVelRhoHat.getLowIndex(),
				     velocityVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(velocityVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.wVelRhoHat.copy(constVelocityVars.wVelocity,
				     velocityVars.wVelRhoHat.getLowIndex(),
				     velocityVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in MomentumSolver for calcVelSrc");

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &velocityVars,
					&constVelocityVars);

#ifdef filter_convection_terms
    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
      switch (index) {
	  case Arches::XDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjU[ii], 
			    d_lab->d_filteredRhoUjULabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::YDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjV[ii], 
			    d_lab->d_filteredRhoUjVLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::ZDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjW[ii], 
			    d_lab->d_filteredRhoUjWLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;
	default:
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeff");
      }
    }

    filterNonlinearTerms(pc, patch, index, cellinfo, &velocityVars);

    IntVector indexLow;
    IntVector indexHigh;
    double areaew, areans, areatb;
	
    switch (index) {
	case Arches::XDIR:
	  indexLow = patch->getSFCXFORTLowIndex();
	  indexHigh = patch->getSFCXFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sewu[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sewu[colX] * cellinfo->sns[colY];

		velocityVars.uVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjU[0])[xplusCell]-
		 (velocityVars.filteredRhoUjU[0])[currCell]) * areaew +
		((velocityVars.filteredRhoUjU[1])[yplusCell]-
		 (velocityVars.filteredRhoUjU[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjU[2])[zplusCell]-
		 (velocityVars.filteredRhoUjU[2])[currCell]) *areatb;
	      }
	    }
	  }
	break;

	case Arches::YDIR:
	  indexLow = patch->getSFCYFORTLowIndex();
	  indexHigh = patch->getSFCYFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->snsv[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->snsv[colY];

		velocityVars.vVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjV[0])[xplusCell]-
		 (velocityVars.filteredRhoUjV[0])[currCell]) * areaew +
		((velocityVars.filteredRhoUjV[1])[yplusCell]-
		 (velocityVars.filteredRhoUjV[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjV[2])[zplusCell]-
		 (velocityVars.filteredRhoUjV[2])[currCell]) * areatb;
	      }
	    }
	  }
	break;

	case Arches::ZDIR:
	  indexLow = patch->getSFCZFORTLowIndex();
	  indexHigh = patch->getSFCZFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stbw[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stbw[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

		velocityVars.wVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjW[0])[xplusCell]-
		 (velocityVars.filteredRhoUjW[0])[currCell]) * areaew +
		((velocityVars.filteredRhoUjW[1])[yplusCell]-
		 (velocityVars.filteredRhoUjW[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjW[2])[zplusCell]-
		 (velocityVars.filteredRhoUjW[2])[currCell]) *areatb;
	      }
	    }
	  }
	break;
	default:
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeff");
    }
#endif

      // for scalesimilarity model add stress tensor to the source of velocity eqn.
      if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
	StencilMatrix<constCCVariable<double> > stressTensor; //9 point tensor
	for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
	  old_dw->get(stressTensor[ii], 
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
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeffPred");
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
      
      d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &velocityVars,
				    &constVelocityVars);

      if (d_boundaryCondition->getIntrusionBC()) {
	d_boundaryCondition->intrusionMomExchangeBC(pc, patch, index,
						    cellinfo, &velocityVars,
						    &constVelocityVars);
	d_boundaryCondition->intrusionVelocityBC(pc, patch, index, 
						 cellinfo, &velocityVars,
						 &constVelocityVars);

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
	d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
					 cellinfo, &velocityVars,
					 &constVelocityVars);
      }

#ifdef Runge_Kutta_3d 
#ifndef Runge_Kutta_3d_ssp
    SFCXVariable<double> temp_uVel;
    SFCYVariable<double> temp_vVel;
    SFCZVariable<double> temp_wVel;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch(index) {

    case Arches::XDIR:

      new_dw->allocateAndPut(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch);
      temp_uVel.initialize(0.0);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              temp_uVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[xshiftedCell])*
	      velocityVars.uVelRhoHat[currCell]-
              (old_density[currCell]+old_density[xshiftedCell])*
              velocityVars.uVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch); */;

    break;

    case Arches::YDIR:

      new_dw->allocateAndPut(temp_vVel, d_lab->d_vVelTempLabel, matlIndex, patch);
      temp_vVel.initialize(0.0);
    
      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              temp_vVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[yshiftedCell])*
	      velocityVars.vVelRhoHat[currCell]-
              (old_density[currCell]+old_density[yshiftedCell])*
              velocityVars.vVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_vVel, d_lab->d_vVelTempLabel, matlIndex, patch); */;
    
    break;

    case Arches::ZDIR:

      new_dw->allocateAndPut(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);
      temp_wVel.initialize(0.0);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              temp_wVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[zshiftedCell])*
	      velocityVars.wVelRhoHat[currCell]-
              (old_density[currCell]+old_density[zshiftedCell])*
              velocityVars.wVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch); */;

    break;

    default:
      throw InvalidValue("Invalid index in Pred MomentumSolver for RK3");
    }
#endif
#endif

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
//    d_boundaryCondition->newrecomputePressureBC(pc, patch, cellinfo,
//						&velocityVars);
    d_boundaryCondition->velRhoHatInletBC(pc, patch, cellinfo,
					  &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatPressureBC(pc, patch, cellinfo,
					     &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatOutletBC(pc, patch, cellinfo, delta_t,
					   &velocityVars, &constVelocityVars);

#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch,&velocityVars,&constVelocityVars);
#endif

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  TAU_PROFILE_STOP(compute);
  }
}





// ****************************************************************************
// Schedule solve of linearized pressure equation, corrector step
// ****************************************************************************
void MomentumSolver::solveVelHatCorr(const LevelP& level,
				     SchedulerP& sched,
				     const int Runge_Kutta_current_step,
				     const bool Runge_Kutta_last_step)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  
#ifdef filter_convection_terms
  sched_computeNonlinearTerms(sched, patches, matls, d_lab, 
			      Runge_Kutta_current_step, Runge_Kutta_last_step);
#endif

  sched_buildLinearMatrixVelHatCorr(sched, patches, matls);

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixVelHatCorr(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeffVelHatCorr", 
			   this, &MomentumSolver::buildLinearMatrixVelHatCorr);
  

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
    tsk->requires(Task::NewDW, d_lab->d_stressTensorCompLabel,
		  d_lab->d_stressTensorMatl,Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
#ifdef Runge_Kutta_3d
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  
#ifdef do_pressure_correction
  tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_viscosityIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  
#ifndef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_uVelTempLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelTempLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelTempLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
#else
#ifndef Runge_Kutta_2nd
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  
#endif
  // from new_dw
  // for new task graph to work
  
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef do_pressure_correction
  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefCorrLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  
  
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
  
#ifdef filter_convection_terms
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjULabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjVLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjWLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
#endif
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity
    
    tsk->computes(d_lab->d_uVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_vVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_wVelRhoHatCorrLabel);
#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_divConstraintLabel);
#endif
    
    sched->addTask(tsk, patches, matls);
}





void 
MomentumSolver::buildLinearMatrixVelHatCorr(const ProcessorGroup* pc,
					    const PatchSubset* patches,
					    const MaterialSubset* /*matls*/,
					    DataWarehouse* old_dw,
					    DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  double zeta_2 = -5.0/12.0;
  delta_t *= gamma_3;
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // compute all three componenets of velocity stencil coefficients

    new_dw->get(constVelocityVars.new_density, d_lab->d_densityCPLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef Runge_Kutta_3d
    new_dw->get(constVelocityVars.density, d_lab->d_densityIntermLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef do_pressure_correction
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressureIntermLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityIntermLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->get(constVelocityVars.density, d_lab->d_densityPredLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef do_pressure_correction
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressurePredLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityPredLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef divergenceconstraint
    new_dw->get(constVelocityVars.scalar, d_lab->d_scalarSPLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.drhodf, d_lab->d_drhodfCPLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constVelocityVars.scalarDiffusionCoeff[ii],
		      d_lab->d_scalDiffCoefCorrLabel,
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(velocityVars.divergence, d_lab->d_divConstraintLabel,
		          matlIndex, patch);
    velocityVars.divergence.initialize(0.0);
#endif
    
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

#ifdef Runge_Kutta_3d
    new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityIntermLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityIntermLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityIntermLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocityIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocityIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocityIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constVelocityVars.old_density, d_lab->d_densityIntermLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		    matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef Runge_Kutta_2nd
    new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocityPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocityPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocityPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constVelocityVars.old_density, d_lab->d_densityPredLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
#endif
    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {
      
      // get multimaterial momentum source terms
      
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
	  throw InvalidValue("invalid index for velocity in MomentumSolver");
	}
      }

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
	new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.uVelRhoHat.copy(constVelocityVars.uVelocity,
				     velocityVars.uVelRhoHat.getLowIndex(),
				     velocityVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(velocityVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.vVelRhoHat.copy(constVelocityVars.vVelocity,
				     velocityVars.vVelRhoHat.getLowIndex(),
				     velocityVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(velocityVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.wVelRhoHat.copy(constVelocityVars.wVelocity,
				     velocityVars.wVelRhoHat.getLowIndex(),
				     velocityVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in MomentumSolver for calcVelSrc");

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &velocityVars,
					&constVelocityVars);

#ifdef filter_convection_terms
    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
      switch (index) {
	  case Arches::XDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjU[ii], 
			    d_lab->d_filteredRhoUjULabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::YDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjV[ii], 
			    d_lab->d_filteredRhoUjVLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::ZDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjW[ii], 
			    d_lab->d_filteredRhoUjWLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;
	default:
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeff");
      }
    }

    filterNonlinearTerms(pc, patch, index, cellinfo, &velocityVars);

    IntVector indexLow;
    IntVector indexHigh;
    double areaew, areans, areatb;
	
    switch (index) {
	case Arches::XDIR:
	  indexLow = patch->getSFCXFORTLowIndex();
	  indexHigh = patch->getSFCXFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sewu[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sewu[colX] * cellinfo->sns[colY];

		velocityVars.uVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjU[0])[xplusCell]-
		 (velocityVars.filteredRhoUjU[0])[currCell]) * areaew +
		((velocityVars.filteredRhoUjU[1])[yplusCell]-
		 (velocityVars.filteredRhoUjU[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjU[2])[zplusCell]-
		 (velocityVars.filteredRhoUjU[2])[currCell]) *areatb;
	      }
	    }
	  }
	break;

	case Arches::YDIR:
	  indexLow = patch->getSFCYFORTLowIndex();
	  indexHigh = patch->getSFCYFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->snsv[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->snsv[colY];

		velocityVars.vVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjV[0])[xplusCell]-
		 (velocityVars.filteredRhoUjV[0])[currCell]) *areaew +
		((velocityVars.filteredRhoUjV[1])[yplusCell]-
		 (velocityVars.filteredRhoUjV[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjV[2])[zplusCell]-
		 (velocityVars.filteredRhoUjV[2])[currCell]) *areatb;
	      }
	    }
	  }
	break;

	case Arches::ZDIR:
	  indexLow = patch->getSFCZFORTLowIndex();
	  indexHigh = patch->getSFCZFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stbw[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stbw[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

		velocityVars.wVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjW[0])[xplusCell]-
		 (velocityVars.filteredRhoUjW[0])[currCell]) *areaew +
		((velocityVars.filteredRhoUjW[1])[yplusCell]-
		 (velocityVars.filteredRhoUjW[1])[currCell]) *areans +
		((velocityVars.filteredRhoUjW[2])[zplusCell]-
		 (velocityVars.filteredRhoUjW[2])[currCell]) *areatb;
	      }
	    }
	  }
	break;
	default:
	  throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeff");
    }
#endif

      // for scalesimilarity model add stress tensor to the source of velocity eqn.
    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
      StencilMatrix<constCCVariable<double> > stressTensor; //9 point tensor
      for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
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
	throw InvalidValue("Invalid index in MomentumSolver::BuildVelCoeffCorr");
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
      
    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &velocityVars,
				    &constVelocityVars);
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
    
    d_discretize->calculateVelDiagonal(pc, patch,
				       index,
				       &velocityVars);


    
    d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &velocityVars,
				     &constVelocityVars);
    
#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
    constSFCXVariable<double> temp_uVel;
    constSFCYVariable<double> temp_vVel;
    constSFCZVariable<double> temp_wVel;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;

    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch(index) {

    case Arches::XDIR:

      new_dw->get(temp_uVel, d_lab->d_uVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              velocityVars.uVelRhoHat[currCell] += zeta_2*temp_uVel[currCell]/
              (0.5*(new_density[currCell]+new_density[xshiftedCell]));
          }
        }
      }
    
    break;

    case Arches::YDIR:

      new_dw->get(temp_vVel, d_lab->d_vVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              velocityVars.vVelRhoHat[currCell] += zeta_2*temp_vVel[currCell]/
              (0.5*(new_density[currCell]+new_density[yshiftedCell]));
          }
        }
      }
    
    break;

    case Arches::ZDIR:

      new_dw->get(temp_wVel, d_lab->d_wVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              velocityVars.wVelRhoHat[currCell] += zeta_2*temp_wVel[currCell]/
              (0.5*(new_density[currCell]+new_density[zshiftedCell]));
          }
        }
      }
    break;

    default:
      throw InvalidValue("Invalid index in Corr MomentumSolver for RK3");
    }
#endif
#endif

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
//    d_boundaryCondition->newrecomputePressureBC(pc, patch,
//						cellinfo, &velocityVars); 
    d_boundaryCondition->velRhoHatInletBC(pc, patch, cellinfo,
					  &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatPressureBC(pc, patch, cellinfo,
					     &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatOutletBC(pc, patch, cellinfo, delta_t,
					   &velocityVars, &constVelocityVars);
#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch,&velocityVars,&constVelocityVars);
#endif


#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}
// ****************************************************************************
// Schedule solve of linearized pressure equation, corrector step
// ****************************************************************************
void MomentumSolver::solveVelHatInterm(const LevelP& level,
				     SchedulerP& sched,
				     const int Runge_Kutta_current_step,
				     const bool Runge_Kutta_last_step)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  
#ifdef filter_convection_terms
  sched_computeNonlinearTerms(sched, patches, matls, d_lab, 
			      Runge_Kutta_current_step, Runge_Kutta_last_step);
#endif

  sched_buildLinearMatrixVelHatInterm(sched, patches, matls);

}
// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixVelHatInterm(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* tsk = scinew Task( "MomentumSolver::BuildCoeffVelHatInterm", 
			   this,&MomentumSolver::buildLinearMatrixVelHatInterm);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
    tsk->requires(Task::NewDW, d_lab->d_stressTensorCompLabel,
		  d_lab->d_stressTensorMatl,Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);


  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
	    	Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef do_pressure_correction
  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_denRefArrayIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef divergenceconstraint
  tsk->requires(Task::NewDW, d_lab->d_scalarIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_drhodfIntermLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefIntermLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
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
    
#ifdef filter_convection_terms
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjULabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjVLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_filteredRhoUjWLabel,
		d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
#endif
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

#ifndef Runge_Kutta_3d_ssp
    tsk->modifies(d_lab->d_uVelTempLabel);
    tsk->modifies(d_lab->d_vVelTempLabel);
    tsk->modifies(d_lab->d_wVelTempLabel);
#endif
    tsk->computes(d_lab->d_uVelRhoHatIntermLabel);
    tsk->computes(d_lab->d_vVelRhoHatIntermLabel);
    tsk->computes(d_lab->d_wVelRhoHatIntermLabel);
#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_divConstraintLabel);
#endif
    
    sched->addTask(tsk, patches, matls);
}



void 
MomentumSolver::buildLinearMatrixVelHatInterm(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  double zeta_1 = -17.0/60.0;
  delta_t *= gamma_2; 
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // compute all three componenets of velocity stencil coefficients


    // Get the reference density
    // Get the required data

    new_dw->get(constVelocityVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constVelocityVars.new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef do_pressure_correction
    new_dw->get(constVelocityVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->get(constVelocityVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayIntermLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    new_dw->get(constVelocityVars.scalar, d_lab->d_scalarIntermLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.drhodf, d_lab->d_drhodfIntermLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constVelocityVars.scalarDiffusionCoeff[ii],
		      d_lab->d_scalDiffCoefIntermLabel,
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(velocityVars.divergence, d_lab->d_divConstraintLabel,
		          matlIndex, patch);
    velocityVars.divergence.initialize(0.0);
#endif

    PerPatch<CellInformationP> cellInfoP;

    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constVelocityVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constVelocityVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

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
	  throw InvalidValue("invalid index for velocity in PressureSolver");
	}
      }

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
	new_dw->allocateAndPut(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.uVelRhoHat.copy(constVelocityVars.uVelocity,
				     velocityVars.uVelRhoHat.getLowIndex(),
				     velocityVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(velocityVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.vVelRhoHat.copy(constVelocityVars.vVelocity,
				     velocityVars.vVelRhoHat.getLowIndex(),
				     velocityVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(velocityVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	velocityVars.wVelRhoHat.copy(constVelocityVars.wVelocity,
				     velocityVars.wVelRhoHat.getLowIndex(),
				     velocityVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in PressureSolver for calcVelSrc");

      }

    d_source->calculateVelocitySource(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars,
				      &constVelocityVars);
    //      d_source->addPressureSource(pc, patch, delta_t, index,
    //				  cellinfo, &velocityVars);

#ifdef filter_convection_terms
    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
      switch (index) {
	  case Arches::XDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjU[ii], 
			    d_lab->d_filteredRhoUjULabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::YDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjV[ii], 
			    d_lab->d_filteredRhoUjVLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;

	case Arches::ZDIR:
	    new_dw->get(constVelocityVars.filteredRhoUjW[ii], 
			    d_lab->d_filteredRhoUjWLabel, ii, patch,
			    Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
	break;
	default:
	  throw InvalidValue("Invalid index in PressureSolver::BuildVelCoeff");
      }
    }

    filterNonlinearTerms(pc, patch, index, cellinfo, &velocityVars);

    IntVector indexLow;
    IntVector indexHigh;
    double areaew, areans, areatb;
	
    switch (index) {
	case Arches::XDIR:
	  indexLow = patch->getSFCXFORTLowIndex();
	  indexHigh = patch->getSFCXFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sewu[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sewu[colX] * cellinfo->sns[colY];

		velocityVars.uVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjU[0])[xplusCell]-(velocityVars.filteredRhoUjU[0])[currCell]) *
		areaew +
		((velocityVars.filteredRhoUjU[1])[yplusCell]-(velocityVars.filteredRhoUjU[1])[currCell]) *
		areans +
		((velocityVars.filteredRhoUjU[2])[zplusCell]-(velocityVars.filteredRhoUjU[2])[currCell]) *
		areatb;
	      }
	    }
	  }
	break;

	case Arches::YDIR:
	  indexLow = patch->getSFCYFORTLowIndex();
	  indexHigh = patch->getSFCYFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->snsv[colY] * cellinfo->stb[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->snsv[colY];

		velocityVars.vVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjV[0])[xplusCell]-(velocityVars.filteredRhoUjV[0])[currCell]) *
		areaew +
		((velocityVars.filteredRhoUjV[1])[yplusCell]-(velocityVars.filteredRhoUjV[1])[currCell]) *
		areans +
		((velocityVars.filteredRhoUjV[2])[zplusCell]-(velocityVars.filteredRhoUjV[2])[currCell]) *
		areatb;
	      }
	    }
	  }
	break;

	case Arches::ZDIR:
	  indexLow = patch->getSFCZFORTLowIndex();
	  indexHigh = patch->getSFCZFORTHighIndex();

	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          	IntVector currCell(colX, colY, colZ);
          	IntVector xplusCell(colX+1, colY, colZ);
          	IntVector yplusCell(colX, colY+1, colZ);
        	IntVector zplusCell(colX, colY, colZ+1);
	  	areaew = cellinfo->sns[colY] * cellinfo->stbw[colZ];
	  	areans = cellinfo->sew[colX] * cellinfo->stbw[colZ];
	  	areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

		velocityVars.wVelNonlinearSrc[currCell] -=
		((velocityVars.filteredRhoUjW[0])[xplusCell]-(velocityVars.filteredRhoUjW[0])[currCell]) *
		areaew +
		((velocityVars.filteredRhoUjW[1])[yplusCell]-(velocityVars.filteredRhoUjW[1])[currCell]) *
		areans +
		((velocityVars.filteredRhoUjW[2])[zplusCell]-(velocityVars.filteredRhoUjW[2])[currCell]) *
		areatb;
	      }
	    }
	  }
	break;
	default:
	  throw InvalidValue("Invalid index in PressureSolver::BuildVelCoeff");
    }
#endif

      // for scalesimilarity model add stress tensor to the source of velocity eqn.
      if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
	StencilMatrix<constCCVariable<double> > stressTensor; //9 point tensor
	for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
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
	  throw InvalidValue("Invalid index in PressureSolver::BuildVelCoeffInterm");
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
      
    d_boundaryCondition->velocityBC(pc, patch, 
                                    index,
				    cellinfo, &velocityVars,
				    &constVelocityVars);
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



    d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &velocityVars,
				     &constVelocityVars);

      
#ifndef Runge_Kutta_3d_ssp
    SFCXVariable<double> temp_uVel;
    SFCYVariable<double> temp_vVel;
    SFCZVariable<double> temp_wVel;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
#ifndef filter_convection_terms
    IntVector indexLow;
    IntVector indexHigh;
#endif

    new_dw->get(old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    switch(index) {

    case Arches::XDIR:

      new_dw->getModifiable(temp_uVel, d_lab->d_uVelTempLabel,
                  matlIndex, patch);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);


              velocityVars.uVelRhoHat[currCell] += zeta_1*temp_uVel[currCell]/
              (0.5*(new_density[currCell]+new_density[xshiftedCell]));
              temp_uVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[xshiftedCell])*
	      velocityVars.uVelRhoHat[currCell]-
              (old_density[currCell]+old_density[xshiftedCell])*
              velocityVars.uVelocity[currCell])/
              gamma_2-zeta_1*temp_uVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);
    
    break;

    case Arches::YDIR:

      new_dw->getModifiable(temp_vVel, d_lab->d_vVelTempLabel,
                  matlIndex, patch);

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              velocityVars.vVelRhoHat[currCell] += zeta_1*temp_vVel[currCell]/
              (0.5*(new_density[currCell]+new_density[yshiftedCell]));
              temp_vVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[yshiftedCell])*
	      velocityVars.vVelRhoHat[currCell]-
              (old_density[currCell]+old_density[yshiftedCell])*
              velocityVars.vVelocity[currCell])/
              gamma_2-zeta_1*temp_vVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch);
    
    break;

    case Arches::ZDIR:

      new_dw->getModifiable(temp_wVel, d_lab->d_wVelTempLabel,
                  matlIndex, patch);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              velocityVars.wVelRhoHat[currCell] += zeta_1*temp_wVel[currCell]/
              (0.5*(new_density[currCell]+new_density[zshiftedCell]));
              temp_wVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[zshiftedCell])*
              velocityVars.wVelRhoHat[currCell]-
              (old_density[currCell]+old_density[zshiftedCell])*
              velocityVars.wVelocity[currCell])/
              gamma_2-zeta_1*temp_wVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);

    break;

    default:
      throw InvalidValue("Invalid index in Interm PressureSolver for RK3");
    }
#endif
    
#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
//    d_boundaryCondition->newrecomputePressureBC(pc, patch,
//    						cellinfo, &velocityVars); 
    d_boundaryCondition->velRhoHatInletBC(pc, patch, cellinfo,
					  &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatPressureBC(pc, patch, cellinfo,
					     &velocityVars, &constVelocityVars);
    d_boundaryCondition->velRhoHatOutletBC(pc, patch, cellinfo, delta_t,
					  &velocityVars, &constVelocityVars);
#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch,&velocityVars,&constVelocityVars);
#endif


  // put required vars

    // allocateAndPut instead:
    /* new_dw->put(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel, 
		matlIndex, patch); */;

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}

//****************************************************************************
// Schedule the averaging of hat velocities for Runge-Kutta step
//****************************************************************************
void 
MomentumSolver::sched_averageRKHatVelocities(SchedulerP& sched,
					 const PatchSet* patches,
				 	 const MaterialSet* matls,
					 const int Runge_Kutta_current_step,
					 const int Runge_Kutta_last_step)
{
  Task* tsk = scinew Task("MomentumSolver::averageRKHatVelocities",
			  this,
			  &MomentumSolver::averageRKHatVelocities,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
                Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (Runge_Kutta_last_step) {
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_uVelRhoHatCorrLabel);
    tsk->modifies(d_lab->d_vVelRhoHatCorrLabel);
    tsk->modifies(d_lab->d_wVelRhoHatCorrLabel);
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_uVelRhoHatIntermLabel);
    tsk->modifies(d_lab->d_vVelRhoHatIntermLabel);
    tsk->modifies(d_lab->d_wVelRhoHatIntermLabel);
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually average the Runge-Kutta hat velocities here
//****************************************************************************
void 
MomentumSolver::averageRKHatVelocities(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* new_dw,
			   const int Runge_Kutta_current_step,
			   const int Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> old_density;
    constCCVariable<double> rho2_density;
    constCCVariable<double> new_density;
    constCCVariable<int> cellType;
    constSFCXVariable<double> old_uvel;
    constSFCYVariable<double> old_vvel;
    constSFCZVariable<double> old_wvel;
    SFCXVariable<double> new_uvel;
    SFCYVariable<double> new_vvel;
    SFCZVariable<double> new_wvel;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(rho2_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_uvel, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_vvel, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_wvel, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (Runge_Kutta_last_step) {
      new_dw->get(new_density, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      new_dw->getModifiable(new_uvel, d_lab->d_uVelRhoHatCorrLabel, 
			    matlIndex, patch);
      new_dw->getModifiable(new_vvel, d_lab->d_vVelRhoHatCorrLabel, 
			    matlIndex, patch);
      new_dw->getModifiable(new_wvel, d_lab->d_wVelRhoHatCorrLabel, 
			    matlIndex, patch);
    }
    else {
      new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      new_dw->getModifiable(new_uvel, d_lab->d_uVelRhoHatIntermLabel, 
			    matlIndex, patch);
      new_dw->getModifiable(new_vvel, d_lab->d_vVelRhoHatIntermLabel, 
			    matlIndex, patch);
      new_dw->getModifiable(new_wvel, d_lab->d_wVelRhoHatIntermLabel, 
			    matlIndex, patch);
    }

    double factor_old, factor_new, factor_divide;
    switch (Runge_Kutta_current_step) {
    case Arches::SECOND:
      if (Runge_Kutta_last_step) {
	factor_old = 1.0;
	factor_new = 1.0;
	factor_divide = 2.0;
      }
      else {
	factor_old = 3.0;
	factor_new = 1.0;
	factor_divide = 4.0;
      }
    break;

    case Arches::THIRD:
	factor_old = 1.0;
	factor_new = 2.0;
	factor_divide = 3.0;
    break;

    default:
      throw InvalidValue("Invalid Runge-Kutta step in averageRKProps");
    }

    IntVector indexLow, indexHigh;
    indexLow = patch->getSFCXFORTLowIndex();
    indexHigh = patch->getSFCXFORTHighIndex();

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector xminusCell(colX-1, colY, colZ);
          
	    new_uvel[currCell] = (factor_old*old_uvel[currCell]*
		(old_density[currCell]+old_density[xminusCell]) +
		factor_new*new_uvel[currCell]*
		(rho2_density[currCell]+rho2_density[xminusCell]))/
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
          
	    new_vvel[currCell] = (factor_old*old_vvel[currCell]*
		(old_density[currCell]+old_density[yminusCell]) +
		factor_new*new_vvel[currCell]*
		(rho2_density[currCell]+rho2_density[yminusCell]))/
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
          
	    new_wvel[currCell] = (factor_old*old_wvel[currCell]*
		(old_density[currCell]+old_density[zminusCell]) +
		factor_new*new_wvel[currCell]*
		(rho2_density[currCell]+rho2_density[zminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[zminusCell]));

	}
      }
    }

  int out_celltypeval = d_boundaryCondition->outletCellType();
  int press_celltypeval = d_boundaryCondition->pressureCellType();
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
        IntVector xminusyminusCell(colX-1, colY-1, colZ);
        IntVector xminuszminusCell(colX-1, colY, colZ-1);

	   new_uvel[currCell] = (factor_old*old_uvel[currCell]*
		(old_density[currCell]+old_density[xminusCell]) +
		factor_new*new_uvel[currCell]*
		(rho2_density[currCell]+rho2_density[xminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[xminusCell]));

 	if ((cellType[xminusCell] == out_celltypeval)||
	    (cellType[xminusCell] == press_celltypeval)) {

           new_uvel[xminusCell] = new_uvel[currCell];

        if (!(yminus && (colY == idxLo.y()))) {
	    new_vvel[xminusCell] = (factor_old*old_vvel[xminusCell]*
		(old_density[xminusCell]+old_density[xminusyminusCell]) +
		factor_new*new_vvel[xminusCell]*
		(rho2_density[xminusCell]+rho2_density[xminusyminusCell]))/
		(factor_divide*
		(new_density[xminusCell]+new_density[xminusyminusCell]));
	}
        if (!(zminus && (colZ == idxLo.z()))) {
	    new_wvel[xminusCell] = (factor_old*old_wvel[xminusCell]*
		(old_density[xminusCell]+old_density[xminuszminusCell]) +
		factor_new*new_wvel[xminusCell]*
		(rho2_density[xminusCell]+rho2_density[xminuszminusCell]))/
		(factor_divide*
		(new_density[xminusCell]+new_density[xminuszminusCell]));
	}
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
        IntVector xplusyminusCell(colX+1, colY-1, colZ);
        IntVector xpluszminusCell(colX+1, colY, colZ-1);
        IntVector xplusplusCell(colX+2, colY, colZ);

	   new_uvel[xplusCell] = (factor_old*old_uvel[xplusCell]*
		(old_density[xplusCell]+old_density[currCell]) +
		factor_new*new_uvel[xplusCell]*
		(rho2_density[xplusCell]+rho2_density[currCell]))/
		(factor_divide*(new_density[xplusCell]+new_density[currCell]));

 	if ((cellType[xplusCell] == out_celltypeval)||
	    (cellType[xplusCell] == press_celltypeval)) {

           new_uvel[xplusplusCell] = new_uvel[xplusCell];

        if (!(yminus && (colY == idxLo.y()))) {
	    new_vvel[xplusCell] = (factor_old*old_vvel[xplusCell]*
		(old_density[xplusCell]+old_density[xplusyminusCell]) +
		factor_new*new_vvel[xplusCell]*
		(rho2_density[xplusCell]+rho2_density[xplusyminusCell]))/
		(factor_divide*
		(new_density[xplusCell]+new_density[xplusyminusCell]));
	}

        if (!(zminus && (colZ == idxLo.z()))) {
	    new_wvel[xplusCell] = (factor_old*old_wvel[xplusCell]*
		(old_density[xplusCell]+old_density[xpluszminusCell]) +
		factor_new*new_wvel[xplusCell]*
		(rho2_density[xplusCell]+rho2_density[xpluszminusCell]))/
		(factor_divide*
		(new_density[xplusCell]+new_density[xpluszminusCell]));
	}
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
        IntVector yminusxminusCell(colX-1, colY-1, colZ);
        IntVector yminuszminusCell(colX, colY-1, colZ-1);

	   new_vvel[currCell] = (factor_old*old_vvel[currCell]*
		(old_density[currCell]+old_density[yminusCell]) +
		factor_new*new_vvel[currCell]*
		(rho2_density[currCell]+rho2_density[yminusCell]))/
		(factor_divide*(new_density[currCell]+new_density[yminusCell]));

 	if ((cellType[yminusCell] == out_celltypeval)||
	    (cellType[yminusCell] == press_celltypeval)) {

           new_vvel[yminusCell] = new_vvel[currCell];

        if (!(xminus && (colX == idxLo.x()))) {
	    new_uvel[yminusCell] = (factor_old*old_uvel[yminusCell]*
		(old_density[yminusCell]+old_density[yminusxminusCell]) +
		factor_new*new_uvel[yminusCell]*
		(rho2_density[yminusCell]+rho2_density[yminusxminusCell]))/
		(factor_divide*
		(new_density[yminusCell]+new_density[yminusxminusCell]));
	}
        if (!(zminus && (colZ == idxLo.z()))) {
	    new_wvel[yminusCell] = (factor_old*old_wvel[yminusCell]*
		(old_density[yminusCell]+old_density[yminuszminusCell]) +
		factor_new*new_wvel[yminusCell]*
		(rho2_density[yminusCell]+rho2_density[yminuszminusCell]))/
		(factor_divide*
		(new_density[yminusCell]+new_density[yminuszminusCell]));
	}
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
        IntVector yplusxminusCell(colX-1, colY+1, colZ);
        IntVector ypluszminusCell(colX, colY+1, colZ-1);
        IntVector yplusplusCell(colX, colY+2, colZ);

	   new_vvel[yplusCell] = (factor_old*old_vvel[yplusCell]*
		(old_density[yplusCell]+old_density[currCell]) +
		factor_new*new_vvel[yplusCell]*
		(rho2_density[yplusCell]+rho2_density[currCell]))/
		(factor_divide*(new_density[yplusCell]+new_density[currCell]));

 	if ((cellType[yplusCell] == out_celltypeval)||
	    (cellType[yplusCell] == press_celltypeval)) {

           new_vvel[yplusplusCell] = new_vvel[yplusCell];

        if (!(xminus && (colX == idxLo.x()))) {
	    new_uvel[yplusCell] = (factor_old*old_uvel[yplusCell]*
		(old_density[yplusCell]+old_density[yplusxminusCell]) +
		factor_new*new_uvel[yplusCell]*
		(rho2_density[yplusCell]+rho2_density[yplusxminusCell]))/
		(factor_divide*
		(new_density[yplusCell]+new_density[yplusxminusCell]));
	}

        if (!(zminus && (colZ == idxLo.z()))) {
	    new_wvel[yplusCell] = (factor_old*old_wvel[yplusCell]*
		(old_density[yplusCell]+old_density[ypluszminusCell]) +
		factor_new*new_wvel[yplusCell]*
		(rho2_density[yplusCell]+rho2_density[ypluszminusCell]))/
		(factor_divide*
		(new_density[yplusCell]+new_density[ypluszminusCell]));
	}
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
        IntVector zminusxminusCell(colX-1, colY, colZ-1);
        IntVector zminusyminusCell(colX, colY-1, colZ-1);

	   new_wvel[currCell] = (factor_old*old_wvel[currCell]*
		(old_density[currCell]+old_density[zminusCell]) +
		factor_new*new_wvel[currCell]*
		(rho2_density[currCell]+rho2_density[zminusCell]))/
		(factor_divide*
		(new_density[currCell]+new_density[zminusCell]));

 	if ((cellType[zminusCell] == out_celltypeval)||
	    (cellType[zminusCell] == press_celltypeval)) {

           new_wvel[zminusCell] = new_wvel[currCell];

        if (!(xminus && (colX == idxLo.x()))) {
	    new_uvel[zminusCell] = (factor_old*old_uvel[zminusCell]*
		(old_density[zminusCell]+old_density[zminusxminusCell]) +
		factor_new*new_uvel[zminusCell]*
		(rho2_density[zminusCell]+rho2_density[zminusxminusCell]))/
		(factor_divide*
		(new_density[zminusCell]+new_density[zminusxminusCell]));
	}
        if (!(yminus && (colY == idxLo.y()))) {
	    new_vvel[zminusCell] = (factor_old*old_vvel[zminusCell]*
		(old_density[zminusCell]+old_density[zminusyminusCell]) +
		factor_new*new_vvel[zminusCell]*
		(rho2_density[zminusCell]+rho2_density[zminusyminusCell]))/
		(factor_divide*
		(new_density[zminusCell]+new_density[zminusyminusCell]));
	}
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
        IntVector zplusxminusCell(colX-1, colY, colZ+1);
        IntVector zplusyminusCell(colX, colY-1, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);

	   new_wvel[zplusCell] = (factor_old*old_wvel[zplusCell]*
		(old_density[zplusCell]+old_density[currCell]) +
		factor_new*new_wvel[zplusCell]*
		(rho2_density[zplusCell]+rho2_density[currCell]))/
		(factor_divide*(new_density[zplusCell]+new_density[currCell]));

 	if ((cellType[zplusCell] == out_celltypeval)||
	    (cellType[zplusCell] == press_celltypeval)) {

           new_wvel[zplusplusCell] = new_wvel[zplusCell];

        if (!(xminus && (colX == idxLo.x()))) {
	    new_uvel[zplusCell] = (factor_old*old_uvel[zplusCell]*
		(old_density[zplusCell]+old_density[zplusxminusCell]) +
		factor_new*new_uvel[zplusCell]*
		(rho2_density[zplusCell]+rho2_density[zplusxminusCell]))/
		(factor_divide*
		(new_density[zplusCell]+new_density[zplusxminusCell]));
	}

        if (!(yminus && (colY == idxLo.y()))) {
	    new_vvel[zplusCell] = (factor_old*old_vvel[zplusCell]*
		(old_density[zplusCell]+old_density[zplusyminusCell]) +
		factor_new*new_vvel[zplusCell]*
		(rho2_density[zplusCell]+rho2_density[zplusyminusCell]))/
		(factor_divide*
		(new_density[zplusCell]+new_density[zplusyminusCell]));
	}
	}
      }
    }
  }
  }
}
