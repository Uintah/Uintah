//----- MomentumSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
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

    // Get the required data

    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(velocityVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.denRefArray, d_lab->d_denRefArrayLabel,
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

    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    new_dw->getCopy(velocityVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // for explicit coeffs will be computed using the old u, v, and w
  // change it for implicit solve
  // get void fraction

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		  matlIndex, patch, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    switch (index) {

    case Arches::XDIR:

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.old_uVelocity, d_lab->d_uVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDU, d_lab->d_DUMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->getCopy(velocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
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

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.old_vVelocity, d_lab->d_vVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    //cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDV, d_lab->d_DVMBLMLabel,
    //			matlIndex, patch);
    // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->getCopy(velocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
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

      new_dw->getCopy(velocityVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(velocityVars.old_wVelocity, d_lab->d_wVelocitySIVBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      //    new_dw->allocate(velocityVars.variableCalledDW, d_lab->d_DWMBLMLabel,
      //			matlIndex, patch);
      // for multimaterial

      if (d_MAlab) {

	new_dw->getCopy(velocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->getCopy(velocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
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
	new_dw->getCopy(velocityVars.uVelocityCoeff[ii], 
		    d_lab->d_uVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->getCopy(velocityVars.uVelNonlinearSrc, 
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
	new_dw->getCopy(velocityVars.vVelocityCoeff[ii], 
		    d_lab->d_vVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->getCopy(velocityVars.vVelNonlinearSrc, 
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
	new_dw->getCopy(velocityVars.wVelocityCoeff[ii], 
		    d_lab->d_wVelCoefMBLMLabel, 
		    ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->getCopy(velocityVars.wVelNonlinearSrc, 
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

    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, 
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

#ifdef Scalar_ENO
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsUPred_label);
#else
    tsk->computes(d_lab->d_maxAbsU_label);
#endif
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

#ifdef Scalar_ENO
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsVPred_label);
#else
    tsk->computes(d_lab->d_maxAbsV_label);
#endif
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

#ifdef Scalar_ENO
#ifdef correctorstep
    tsk->computes(d_lab->d_maxAbsWPred_label);
#else
    tsk->computes(d_lab->d_maxAbsW_label);
#endif
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

    // Get the required data
#ifdef correctorstep
    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#else
    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityCPLabel, 
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

    if (d_MAlab) {

      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		      matlIndex, patch, 
		      Ghost::AroundCells, Arches::ONEGHOSTCELL);

      new_dw->getCopy(velocityVars.cellType, d_lab->d_mmcellTypeLabel,
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
						    delta_t, index,
						    cellinfo, &velocityVars);

    }
    else {
    
      d_source->calculateVelocityPred(pc, patch, 
				      delta_t, index,
				      cellinfo, &velocityVars);

    }

  #ifdef Scalar_ENO
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
  #endif

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
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_2nd
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif 
  #ifdef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif 

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatCorrLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_2nd
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_uVelocitySPBCLabel);

#ifdef Scalar_ENO
    tsk->computes(d_lab->d_maxAbsU_label);
#endif

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_2nd
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_vVelocitySPBCLabel);

#ifdef Scalar_ENO
    tsk->computes(d_lab->d_maxAbsV_label);
#endif

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatCorrLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_2nd
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_wVelocitySPBCLabel);

#ifdef Scalar_ENO
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

    // Get the required data

    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityCPLabel, 
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

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
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
				    cellinfo, &velocityVars);
  #ifdef Runge_Kutta_2nd
    constSFCXVariable<double> old_uVelocity;
    constSFCYVariable<double> old_vVelocity;
    constSFCZVariable<double> old_wVelocity;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;
    
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch (index) {
    case Arches::XDIR:

      new_dw->get(old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              velocityVars.uVelRhoHat[currCell] = 
              (velocityVars.uVelRhoHat[currCell]+
               (old_density[currCell]+old_density[xshiftedCell])/
               (new_density[currCell]+new_density[xshiftedCell])*
	       old_uVelocity[currCell])/2.0;
          }
        }
      }

      break;
    case Arches::YDIR:

      new_dw->get(old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              velocityVars.vVelRhoHat[currCell] = 
              (velocityVars.vVelRhoHat[currCell]+
               (old_density[currCell]+old_density[yshiftedCell])/
               (new_density[currCell]+new_density[yshiftedCell])*
	       old_vVelocity[currCell])/2.0;
          }
        }
      }

      break;
    case Arches::ZDIR:

      new_dw->get(old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              velocityVars.wVelRhoHat[currCell] = 
              (velocityVars.wVelRhoHat[currCell]+
               (old_density[currCell]+old_density[zshiftedCell])/
               (new_density[currCell]+new_density[zshiftedCell])*
	       old_wVelocity[currCell])/2.0;
          }
        }
      }

      break;
    default:
      throw InvalidValue("Invalid index in RK2 step");
    }
  #endif
  #ifdef Runge_Kutta_3d_ssp
    constSFCXVariable<double> old_uVelocity;
    constSFCYVariable<double> old_vVelocity;
    constSFCZVariable<double> old_wVelocity;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;
    
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch (index) {
    case Arches::XDIR:

      new_dw->get(old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              velocityVars.uVelRhoHat[currCell] = 
              (2.0*velocityVars.uVelRhoHat[currCell]+
               (old_density[currCell]+old_density[xshiftedCell])/
               (new_density[currCell]+new_density[xshiftedCell])*
	       old_uVelocity[currCell])/3.0;
          }
        }
      }

      break;
    case Arches::YDIR:

      new_dw->get(old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              velocityVars.vVelRhoHat[currCell] = 
              (2.0*velocityVars.vVelRhoHat[currCell]+
               (old_density[currCell]+old_density[yshiftedCell])/
               (new_density[currCell]+new_density[yshiftedCell])*
	       old_vVelocity[currCell])/3.0;
          }
        }
      }

      break;
    case Arches::ZDIR:

      new_dw->get(old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              velocityVars.wVelRhoHat[currCell] = 
              (2.0*velocityVars.wVelRhoHat[currCell]+
               (old_density[currCell]+old_density[zshiftedCell])/
               (new_density[currCell]+new_density[zshiftedCell])*
	       old_wVelocity[currCell])/3.0;
          }
        }
      }

      break;
    default:
      throw InvalidValue("Invalid index in RK3 step");
    }
  #endif

  #ifdef Scalar_ENO
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
  #endif

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
  
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif 

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  switch (index) {

  case Arches::XDIR:

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_uVelocityIntermLabel);

#ifdef Scalar_ENO
    tsk->computes(d_lab->d_maxAbsUInterm_label);
#endif

    break;

  case Arches::YDIR:

    // use new uvelocity for v coef calculation
    
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_vVelocityIntermLabel);

#ifdef Scalar_ENO
    tsk->computes(d_lab->d_maxAbsVInterm_label);
#endif

    break;

  case Arches::ZDIR:

    // use new uvelocity for v coef calculation

    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif 
    tsk->computes(d_lab->d_wVelocityIntermLabel);

#ifdef Scalar_ENO
    tsk->computes(d_lab->d_maxAbsWInterm_label);
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

    // Get the required data

    new_dw->getCopy(velocityVars.pressure, d_lab->d_pressureIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(velocityVars.old_density, d_lab->d_densityIntermLabel, 
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

    if (d_MAlab)
      new_dw->getCopy(velocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,
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
				    cellinfo, &velocityVars);
  #ifdef Runge_Kutta_3d_ssp
    constSFCXVariable<double> old_uVelocity;
    constSFCYVariable<double> old_vVelocity;
    constSFCZVariable<double> old_wVelocity;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;
    
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch (index) {
    case Arches::XDIR:

      new_dw->get(old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              velocityVars.uVelRhoHat[currCell] = 
              (velocityVars.uVelRhoHat[currCell]+
               (old_density[currCell]+old_density[xshiftedCell])/
               (new_density[currCell]+new_density[xshiftedCell])*
	       3.0*old_uVelocity[currCell])/4.0;
          }
        }
      }

      break;
    case Arches::YDIR:

      new_dw->get(old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              velocityVars.vVelRhoHat[currCell] = 
              (velocityVars.vVelRhoHat[currCell]+
               (old_density[currCell]+old_density[yshiftedCell])/
               (new_density[currCell]+new_density[yshiftedCell])*
	       3.0*old_vVelocity[currCell])/4.0;
          }
        }
      }

      break;
    case Arches::ZDIR:

      new_dw->get(old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              velocityVars.wVelRhoHat[currCell] = 
              (velocityVars.wVelRhoHat[currCell]+
               (old_density[currCell]+old_density[zshiftedCell])/
               (new_density[currCell]+new_density[zshiftedCell])*
	       3.0*old_wVelocity[currCell])/4.0;
          }
        }
      }

      break;
    default:
      throw InvalidValue("Invalid index in RK3 step");
    }
  #endif

  #ifdef Scalar_ENO
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
  #endif

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

