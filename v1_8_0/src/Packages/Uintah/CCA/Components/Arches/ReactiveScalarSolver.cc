//----- ReactiveScalarSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ReactiveScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
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
// Default constructor for ReactiveScalarSolver
//****************************************************************************
ReactiveScalarSolver::ReactiveScalarSolver(const ArchesLabel* label,
			   const MPMArchesLabel* MAlb,
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
ReactiveScalarSolver::~ReactiveScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ReactiveScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ReactiveScalarSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
    //throw InvalidValue("Finite Differencing scheme "
	//	       "not supported: " + finite_diff, db);
  }
  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"l2up");
//  if (db->findBlock("convection_scheme")) {
//    db->require("convection_scheme",conv_scheme);
    if (conv_scheme == "l2up") d_conv_scheme = 0;
    else if (conv_scheme == "eno") d_conv_scheme = 1;
         else if (conv_scheme == "weno") d_conv_scheme = 2;
	      else throw InvalidValue("Convection scheme "
		       "not supported: " + conv_scheme);
//  } else
//    d_conv_scheme = 0;
  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs")
    d_linearSolver = scinew RBGSSolver();
  else if (linear_sol == "petsc")
     d_linearSolver = scinew PetscSolver(0); // CHEAT - steve d_myworld);
  else {
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
  }
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule solve of linearized reactscalar equation
//****************************************************************************
void 
ReactiveScalarSolver::solvePred(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : reactscalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixPred(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, reactscalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, reactscalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_reactscalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_reactscalarLinearSolvePred(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrixPred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::BuildCoeffPred",
			  this,
			  &ReactiveScalarSolver::buildLinearMatrixPred,
			  index);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires reactscalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (d_conv_scheme > 0) {
    tsk->requires(Task::OldDW, d_lab->d_maxAbsU_label);
    tsk->requires(Task::OldDW, d_lab->d_maxAbsV_label);
    tsk->requires(Task::OldDW, d_lab->d_maxAbsW_label);
  }

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_reactscalCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalDiffCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalNonLinSrcPredLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrixPred(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
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
  
  double maxAbsU;
  double maxAbsV;
  double maxAbsW;
  if (d_conv_scheme > 0) {
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    old_dw->get(mxAbsU, d_lab->d_maxAbsU_label);
    old_dw->get(mxAbsV, d_lab->d_maxAbsV_label);
    old_dw->get(mxAbsW, d_lab->d_maxAbsW_label);
    maxAbsU = mxAbsU;
    maxAbsV = mxAbsW;
    maxAbsW = mxAbsW;
  }

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    //    new_dw->get(reactscalarVars.old_scalar, d_lab->d_reactscalarINLabel, 
    //		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(constReactscalarVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    // for explicit get old values
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // computes reaction scalar source term in properties
    old_dw->get(constReactscalarVars.reactscalarSRC,
                    d_lab->d_reactscalarSRCINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(reactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefPredLabel, ii, patch);
      new_dw->allocateTemporary(reactscalarVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(reactscalarVars.scalarDiffusionCoeff[ii], d_lab->d_reactscalDiffCoefPredLabel, ii, patch);
    }
    new_dw->allocateTemporary(reactscalarVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(reactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcPredLabel, matlIndex, patch);
 
  // compute ith component of reactscalar stencil coefficients
  // inputs : reactscalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &reactscalarVars, &constReactscalarVars,
				       d_conv_scheme);

    // Calculate reactscalar source terms
    // inputs : [u,v,w]VelocityMS, reactscalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wallID = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &reactscalarVars,
						&constReactscalarVars, wallID);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &reactscalarVars,
					       &constReactscalarVars, wallID);
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, reactscalCoefSBLM
    // outputs: reactscalCoefSBLM
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &reactscalarVars, &constReactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
				&reactscalarVars, &constReactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index,
				     &reactscalarVars, &constReactscalarVars,
				     d_conv_scheme);
    
    // Calculate the reactscalar diagonal terms
    // inputs : reactscalCoefSBLM, scalLinSrcSBLM
    // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &reactscalarVars);

    // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &reactscalarVars,
					    &constReactscalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolvePred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::reactscalarLinearSolvePred",
			  this,
			  &ReactiveScalarSolver::reactscalarLinearSolvePred,
			  index);
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#ifdef correctorstep
  tsk->computes(d_lab->d_reactscalarPredLabel);
#else
  tsk->computes(d_lab->d_reactscalarSPLabel);
#endif
  
#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  tsk->computes(d_lab->d_reactscalarTempLabel);
#endif
#endif

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolvePred(const ProcessorGroup* pc,
                                const PatchSubset* patches,
				const MaterialSubset*,
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
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    //DataWarehouseP old_dw = new_dw->getTop();
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // get old_dw from getTop function
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar,d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit calculation
#ifdef correctorstep
    new_dw->allocateAndPut(reactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
                matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->allocateAndPut(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
                matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->copyOut(reactscalarVars.scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constReactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefPredLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcPredLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(reactscalarVars.residualReactivescalar,  patch);

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
                                  &reactscalarVars, &constReactscalarVars,
				  cellinfo, d_lab);
				  // put back the results
#if 0
    cerr << "print reactscalar solve after predict" << endl;
    reactscalarVars.scalar.print(cerr);
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
    CCVariable<double> temp_reactscalar;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(temp_reactscalar, d_lab->d_reactscalarTempLabel, matlIndex, patch);
    temp_reactscalar.initialize(0.0);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            temp_reactscalar[currCell] = old_density[currCell]*
	    (reactscalarVars.scalar[currCell]-
            reactscalarVars.old_scalar[currCell])/gamma_1;
        }
      }
    }
    // allocateAndPut instead:
    /* new_dw->put(temp_reactscalar, d_lab->d_reactscalarTempLabel, matlIndex, patch); */;
#endif
#endif

// Outlet bc is done here not to change old scalar
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    d_boundaryCondition->scalarOutletBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars, &constReactscalarVars,
					delta_t);

    d_boundaryCondition->scalarPressureBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars,&constReactscalarVars);
#ifdef correctorstep
    // allocateAndPut instead:
    /* new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
                matlIndex, patch); */;
#else
    // allocateAndPut instead:
    /* new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
                matlIndex, patch); */;
#endif

  }
}

//****************************************************************************
// Schedule solve of linearized reactscalar equation, corrector step
//****************************************************************************
void 
ReactiveScalarSolver::solveCorr(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixCorr(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_reactscalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_reactscalarLinearSolveCorr(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrixCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::BuildCoeffCorr",
			  this,
			  &ReactiveScalarSolver::buildLinearMatrixCorr,
			  index);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires reactscalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
//  tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel,
//		Ghost::AroundCells, Arches::ONEGHOSTCELL);
//  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
//		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarIntermLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_reactscalarSRCINIntermLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
  #ifndef Runge_Kutta_2nd
  //tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel,
  //		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_reactscalarSRCINPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif

  if (d_conv_scheme > 0) {
  #ifdef Runge_Kutta_3d
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUInterm_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVInterm_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWInterm_label);
  #else
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWPred_label);
  #endif
  }

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_reactscalCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalDiffCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
 
  tsk->computes(d_lab->d_reactscalNonLinSrcCorrLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrixCorr(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
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
  delta_t *= gamma_3;
#endif
#endif
  
  double maxAbsU;
  double maxAbsV;
  double maxAbsW;
  if (d_conv_scheme > 0) {
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
#ifdef Runge_Kutta_3d
    new_dw->get(mxAbsU, d_lab->d_maxAbsUInterm_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVInterm_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWInterm_label);
#else
    new_dw->get(mxAbsU, d_lab->d_maxAbsUPred_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVPred_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWPred_label);
#endif
    maxAbsU = mxAbsU;
    maxAbsV = mxAbsW;
    maxAbsW = mxAbsW;
  }

  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // from new_dw get DEN, VIS, F(index), U, V, W
  #ifdef Runge_Kutta_3d
    // old_density and old_reactscalar for Runge-Kutta are NOT from initial timestep
    // but from previous (Interm) time step
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  #else
  #ifdef Runge_Kutta_2nd
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    new_dw->get(constReactscalarVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  #endif
    // for explicit get old values
  #ifdef Runge_Kutta_3d
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // computes reaction scalar source term in properties
    new_dw->get(constReactscalarVars.reactscalarSRC,
                    d_lab->d_reactscalarSRCINIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // computes reaction scalar source term in properties
    new_dw->get(constReactscalarVars.reactscalarSRC,
                    d_lab->d_reactscalarSRCINPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(reactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefCorrLabel, ii, patch);
      new_dw->allocateTemporary(reactscalarVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(reactscalarVars.scalarDiffusionCoeff[ii], d_lab->d_reactscalDiffCoefCorrLabel, ii, patch);

    }
    new_dw->allocateTemporary(reactscalarVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(reactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcCorrLabel, matlIndex, patch);
 
  // compute ith component of scalar stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &reactscalarVars, &constReactscalarVars,
				       d_conv_scheme);

    // Calculate scalar source terms
    // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wallID = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &reactscalarVars,
						&constReactscalarVars, wallID);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &reactscalarVars,
					       &constReactscalarVars, wallID);
    }

    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &reactscalarVars, &constReactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
				&reactscalarVars, &constReactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index,
				     &reactscalarVars, &constReactscalarVars,
				     d_conv_scheme);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &reactscalarVars);

    // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &reactscalarVars,
					    &constReactscalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolveCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::reactscalarLinearSolveCorr",
			  this,
			  &ReactiveScalarSolver::reactscalarLinearSolveCorr,
			  index);
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred  
  #ifdef Runge_Kutta_3d
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #ifndef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_reactscalarTempLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  #else
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefCorrLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcCorrLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->computes(d_lab->d_reactscalarSPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolveCorr(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
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
  delta_t *= gamma_3;
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // get old_dw from getTop function
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    // ***warning* 21st July changed from IN to Pred
  #ifdef Runge_Kutta_3d
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar,d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    // for explicit calculation
    new_dw->allocateAndPut(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
    new_dw->copyOut(reactscalarVars.scalar, d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
    new_dw->copyOut(reactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constReactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefCorrLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcCorrLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(reactscalarVars.residualReactivescalar,  patch);

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
                                  &reactscalarVars, &constReactscalarVars,
				  cellinfo, d_lab);
  #ifdef Runge_Kutta_3d
  #ifndef Runge_Kutta_3d_ssp
    constCCVariable<double> temp_reactscalar;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(temp_reactscalar, d_lab->d_reactscalarTempLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            reactscalarVars.scalar[currCell] +=
	    zeta_2*temp_reactscalar[currCell]/
            old_density[currCell];
            if (reactscalarVars.scalar[currCell] > 1.0) 
		reactscalarVars.scalar[currCell] = 1.0;
            else if (reactscalarVars.scalar[currCell] < 0.0)
            	reactscalarVars.scalar[currCell] = 0.0;
        }
      }
    }
  #endif
  #endif

// Outlet bc is done here not to change old scalar
  #ifdef Runge_Kutta_3d
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    d_boundaryCondition->scalarOutletBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars, &constReactscalarVars,
					delta_t);

    d_boundaryCondition->scalarPressureBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars,&constReactscalarVars);
  // put back the results
    // allocateAndPut instead:
    /* new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch); */;
  }
}

//****************************************************************************
// Schedule solve of linearized scalar equation, intermediate step
//****************************************************************************
void 
ReactiveScalarSolver::solveInterm(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixInterm(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_reactscalarLinearSolveInterm(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrixInterm(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::BuildCoeffInterm",
			  this,
			  &ReactiveScalarSolver::buildLinearMatrixInterm,
			  index);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
//  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel,
//		Ghost::AroundCells, Arches::ONEGHOSTCELL);
//  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
//		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_reactscalarSRCINPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_conv_scheme > 0) {
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWPred_label);
  }

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_reactscalCoefIntermLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalDiffCoefIntermLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalNonLinSrcIntermLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrixInterm(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw,
					 int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  delta_t *= gamma_2; 
#endif
  
  double maxAbsU;
  double maxAbsV;
  double maxAbsW;
  if (d_conv_scheme > 0) {
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    new_dw->get(mxAbsU, d_lab->d_maxAbsUPred_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVPred_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWPred_label);
    maxAbsU = mxAbsU;
    maxAbsV = mxAbsW;
    maxAbsW = mxAbsW;
  }
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(constReactscalarVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    // for explicit get old values
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // computes reaction scalar source term in properties
    new_dw->get(constReactscalarVars.reactscalarSRC,
                    d_lab->d_reactscalarSRCINPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(reactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefIntermLabel, ii, patch);
      new_dw->allocateTemporary(reactscalarVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(reactscalarVars.scalarDiffusionCoeff[ii], d_lab->d_reactscalDiffCoefIntermLabel, ii, patch);
    }
    new_dw->allocateTemporary(reactscalarVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(reactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcIntermLabel, matlIndex, patch);
 
  // compute ith component of scalar stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &reactscalarVars, &constReactscalarVars,
				       d_conv_scheme);

    // Calculate scalar source terms
    // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wallID = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &reactscalarVars,
						&constReactscalarVars, wallID);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &reactscalarVars,
					       &constReactscalarVars, wallID);
    }

    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &reactscalarVars, &constReactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
				&reactscalarVars, &constReactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index,
				     &reactscalarVars, &constReactscalarVars,
				     d_conv_scheme);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &reactscalarVars);

    // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &reactscalarVars,
					    &constReactscalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolveInterm(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::reactscalarLinearSolveInterm",
			  this,
			  &ReactiveScalarSolver::reactscalarLinearSolveInterm,
			  index);
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred  
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefIntermLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcIntermLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #ifndef Runge_Kutta_3d_ssp
  tsk->modifies(d_lab->d_reactscalarTempLabel);
  #endif
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->computes(d_lab->d_reactscalarIntermLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolveInterm(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
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
  delta_t *= gamma_2; 
#endif
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // get old_dw from getTop function
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit calculation
    new_dw->allocateAndPut(reactscalarVars.scalar, d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->copyOut(reactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constReactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefIntermLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcIntermLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(reactscalarVars.residualScalar,  patch);

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
                                  &reactscalarVars, &constReactscalarVars,
				  cellinfo, d_lab);

  #ifndef Runge_Kutta_3d_ssp
    CCVariable<double> temp_reactscalar;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(temp_reactscalar, d_lab->d_reactscalarTempLabel,
                matlIndex, patch);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            reactscalarVars.scalar[currCell] +=
	    zeta_1*temp_reactscalar[currCell]/
            old_density[currCell];
            temp_reactscalar[currCell] = old_density[currCell]*
	    (reactscalarVars.scalar[currCell]-
            reactscalarVars.old_scalar[currCell])/
            gamma_2-zeta_1*temp_reactscalar[currCell]/gamma_2;
            if (reactscalarVars.scalar[currCell] > 1.0) 
		reactscalarVars.scalar[currCell] = 1.0;
            else if (reactscalarVars.scalar[currCell] < 0.0)
            	reactscalarVars.scalar[currCell] = 0.0;
        }
      }
    } 
//  new_dw->put(temp_reactscalar, d_lab->d_reactscalarTempLabel, matlIndex, patch);
  #endif
  
// Outlet bc is done here not to change old scalar
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    d_boundaryCondition->scalarOutletBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars, &constReactscalarVars,
					delta_t);

    d_boundaryCondition->scalarPressureBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars,&constReactscalarVars);
    // put back the results
    // allocateAndPut instead:
    /* new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarIntermLabel, 
		matlIndex, patch); */;
  }
}
