//----- PicardNonlinearSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PicardNonlinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ScalarSolver.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

#include <math.h>

using namespace Uintah;

// ****************************************************************************
// Default constructor for PicardNonlinearSolver
// ****************************************************************************
PicardNonlinearSolver::
PicardNonlinearSolver(const ArchesLabel* label, 
		      Properties* props, 
		      BoundaryCondition* bc,
		      TurbulenceModel* turbModel,
		      PhysicalConstants* physConst,
		      const ProcessorGroup* myworld): NonlinearSolver(myworld),
		       d_lab(label), d_props(props), 
		       d_boundaryCondition(bc), d_turbModel(turbModel),
		       d_physicalConsts(physConst)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PicardNonlinearSolver::~PicardNonlinearSolver()
{
}

// ****************************************************************************
// Problem Setup 
// ****************************************************************************
void 
PicardNonlinearSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PicardSolver");
  db->require("max_iter", d_nonlinear_its);
  
  // ** WARNING ** temporarily commented out
  // dw->put(nonlinear_its, "max_nonlinear_its");

  db->require("res_tol", d_resTol);
  bool calPress;
  db->require("cal_pressure", calPress);
  if (calPress) {
    d_pressSolver = scinew PressureSolver(d_lab,
					  d_turbModel, d_boundaryCondition,
					  d_physicalConsts, d_myworld);
    d_pressSolver->problemSetup(db);
  }
  bool calMom;
  db->require("cal_momentum", calMom);
  if (calMom) {
    d_momSolver = scinew MomentumSolver(d_lab, d_turbModel, d_boundaryCondition,
				     d_physicalConsts);
    d_momSolver->problemSetup(db);
  }
  bool calScalar;
  db->require("cal_mixturescalar", calScalar);
  if (calScalar) {
    d_scalarSolver = scinew ScalarSolver(d_lab, d_turbModel, d_boundaryCondition,
				      d_physicalConsts);
    d_scalarSolver->problemSetup(db);
  }
}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int PicardNonlinearSolver::nonlinearSolve(const LevelP& level,
					  SchedulerP& sched,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  double time, double delta_t)
{
  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP, densityCP,
  //                     viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN
  sched_setInitialGuess(level, sched, old_dw, new_dw);

  // Start the iterations
  int nlIterations = 0;
  double nlResidual = 2.0*d_resTol;;
  int nofScalars = d_props->getNumMixVars();
  do{
    //correct inlet velocities to account for change in properties
    // require : densityIN, [u,v,w]VelocityIN (new_dw)
    // compute : [u,v,w]VelocitySIVBC
    d_boundaryCondition->sched_setInletVelocityBC(level, sched, old_dw, new_dw);

    // linearizes and solves pressure eqn
    // require : pressureIN, densityIN, viscosityIN,
    //           [u,v,w]VelocitySIVBC (new_dw)
    //           [u,v,w]VelocitySPBC, densityCP (old_dw)
    // compute : [u,v,w]VelConvCoefPBLM, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM, (matrix_dw)
    //           presResidualPS, presCoefPBLM, presNonLinSrcPBLM,(matrix_dw)
    //           pressurePS (new_dw)
    d_pressSolver->solve(level, sched, old_dw, new_dw, time, delta_t);


    // if external boundary then recompute velocities using new pressure
    // and puts them in nonlinear_dw
    // require : densityCP, pressurePS, [u,v,w]VelocitySIVBC
    // compute : [u,v,w]VelocityCPBC, pressureSPBC
    d_boundaryCondition->sched_recomputePressureBC(level, sched, old_dw,
						 new_dw);

    // Momentum solver
    // require : pressureSPBC, [u,v,w]VelocityCPBC, densityIN, 
    // viscosityIN (new_dw)
    //           [u,v,w]VelocitySPBC, densityCP (old_dw)
    // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //           [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
    //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
    //           [u,v,w]VelocitySPBC
    for (int index = 1; index <= Arches::NDIM; ++index) {
      d_momSolver->solve(level, sched, old_dw, new_dw, time, delta_t, index);
    }
    // equation for scalars
    // require : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN (new_dw)
    //           scalarSP, densityCP (old_dw)
    // compute : scalarCoefSBLM, scalarLinSrcSBLM, scalarNonLinSrcSBLM
    //           scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSS
    for (int index = 0;index < nofScalars; index ++) {
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_scalarSolver->solve(level, sched, old_dw, new_dw, time, delta_t, index);
    }


    // update properties
    // require : densityIN
    // compute : densityCP
    d_props->sched_reComputeProps(level, sched, old_dw, new_dw);

    // LES Turbulence model to compute turbulent viscosity
    // that accounts for sub-grid scale turbulence
    // require : densityCP, viscosityIN, [u,v,w]VelocitySPBC
    // compute : viscosityCTS
    d_turbModel->sched_reComputeTurbSubmodel(level, sched, old_dw, new_dw);


    ++nlIterations;
#if 0    
    // residual represents the degrees of inaccuracies
    nlResidual = computeResidual(level, sched, old_dw, new_dw);
#endif
  }while((nlIterations < d_nonlinear_its)&&(nlResidual > d_resTol));

  // Schedule an interpolation of the face centered velocity data 
  // to a cell centered vector for used by the viz tools
  sched_interpolateFromFCToCC(level, sched, old_dw, new_dw);

  /* Do 'save's in the DataArchiver section of the problem specification now

  // Save the old data (previous time step)
#if 0
  new_dw->pleaseSave(d_lab->d_pressureINLabel, 1);
  new_dw->pleaseSave(d_lab->d_uVelocityINLabel, 1);
  new_dw->pleaseSave(d_lab->d_vVelocityINLabel, 1);
  new_dw->pleaseSave(d_lab->d_wVelocityINLabel, 1);
  new_dw->pleaseSave(d_lab->d_scalarINLabel, nofScalars);
  new_dw->pleaseSave(d_lab->d_densityINLabel, 1);
  new_dw->pleaseSave(d_lab->d_viscosityINLabel, 1);
#endif

  // Save the old velocity as a CC<Vector> Variable
  new_dw->pleaseSave(d_lab->d_oldCCVelocityLabel, 1);

  // Save the new data (this time step)
  new_dw->pleaseSave(d_lab->d_pressurePSLabel, 1);
  new_dw->pleaseSave(d_lab->d_uVelocitySPBCLabel, 1);
  new_dw->pleaseSave(d_lab->d_vVelocitySPBCLabel, 1);
  new_dw->pleaseSave(d_lab->d_wVelocitySPBCLabel, 1);
  new_dw->pleaseSave(d_lab->d_scalarSPLabel, nofScalars);
  new_dw->pleaseSave(d_lab->d_densityCPLabel, 1);
  new_dw->pleaseSave(d_lab->d_viscosityCTSLabel, 1);

  // Save the new velocity as a CC<Vector> Variable
  new_dw->pleaseSave(d_lab->d_newCCVelocityLabel, 1);
  */
  
  return(0);
}

// ****************************************************************************
// Schedule initialize 
// ****************************************************************************
void 
PicardNonlinearSolver::sched_setInitialGuess(const LevelP& level,
					     SchedulerP& sched,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = scinew Task("PicardNonlinearSolver::initialGuess",patch,
			   old_dw, new_dw, this,
			   &PicardNonlinearSolver::setInitialGuess);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

      int nofScalars = d_props->getNumMixVars();
      for (int ii = 0; ii < nofScalars; ii++) {
	tsk->requires(old_dw, d_lab->d_scalarSPLabel, ii, patch, 
		      Ghost::None, numGhostCells);
      }
      tsk->requires(old_dw, d_lab->d_densityCPLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->computes(new_dw, d_lab->d_cellTypeLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_cellInfoLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_pressureINLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_uVelocityINLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelocityINLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelocityINLabel, matlIndex, patch);
      for (int ii = 0; ii < nofScalars; ii++) {
	tsk->computes(new_dw, d_lab->d_scalarINLabel, ii, patch);
      }
      tsk->computes(new_dw, d_lab->d_densityINLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
PicardNonlinearSolver::sched_interpolateFromFCToCC(const LevelP& level,
						   SchedulerP& sched,
						   DataWarehouseP& old_dw,
						   DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PicardNonlinearSolver::interpolateFCToCC",patch,
			   old_dw, new_dw, this,
			   &PicardNonlinearSolver::interpolateFromFCToCC);
      int numGhostCells = 1;
      int matlIndex = 0;

      tsk->requires(new_dw, d_lab->d_uVelocityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelocityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelocityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);

      tsk->computes(new_dw, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_newCCVelocityLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}
// ****************************************************************************
// Actual initialize 
// ****************************************************************************
void 
PicardNonlinearSolver::setInitialGuess(const ProcessorGroup* ,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  int matlIndex = 0;
  int nofGhostCells = 0;
  CCVariable<int> cellType;
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
	      Ghost::None, nofGhostCells);
  CCVariable<double> pressure;
  old_dw->get(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);

  SFCXVariable<double> uVelocity;
  old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);
  SFCYVariable<double> vVelocity;
  old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);
  SFCZVariable<double> wVelocity;
  old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);

  int nofScalars = d_props->getNumMixVars();
  vector<CCVariable<double> > scalar(nofScalars);
  for (int ii = 0; ii < nofScalars; ii++) {
    old_dw->get(scalar[ii], d_lab->d_scalarSPLabel, ii, patch, 
		Ghost::None, nofGhostCells);
  }

  CCVariable<double> density;
  old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);


  // Create vars for new_dw ***warning changed new_dw to old_dw...check
  CCVariable<int> cellType_new;
  new_dw->allocate(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch);
  cellType_new = cellType;
    // Get the PerPatch CellInformation data
  PerPatch<CellInformationP> cellInfoP;
  if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  else {
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  }

#if 0
  PerPatch<CellInformationP> cellInfoP;
  cellInfoP.setData(scinew CellInformation(patch));
  new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
#endif
  CCVariable<double> pressure_new;
  new_dw->allocate(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch);
  pressure_new = pressure; // copy old into new

  SFCXVariable<double> uVelocity_new;
  new_dw->allocate(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch);
  uVelocity_new = uVelocity; // copy old into new
  SFCYVariable<double> vVelocity_new;
  new_dw->allocate(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch);
  vVelocity_new = vVelocity; // copy old into new
  SFCZVariable<double> wVelocity_new;
  new_dw->allocate(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch);
  wVelocity_new = wVelocity; // copy old into new

  vector<CCVariable<double> > scalar_new(nofScalars);
  for (int ii = 0; ii < nofScalars; ii++) {
    new_dw->allocate(scalar_new[ii], d_lab->d_scalarINLabel, ii, patch);
    scalar_new[ii] = scalar[ii]; // copy old into new
  }

  CCVariable<double> density_new;
  new_dw->allocate(density_new, d_lab->d_densityINLabel, matlIndex, patch);
  density_new = density; // copy old into new

  CCVariable<double> viscosity_new;
  new_dw->allocate(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch);
  viscosity_new = viscosity; // copy old into new

  // Copy the variables into the new datawarehouse
  new_dw->put(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch);
  new_dw->put(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch);
  new_dw->put(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch);
  new_dw->put(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch);
  new_dw->put(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch);
  for (int ii = 0; ii < nofScalars; ii++) {
    new_dw->put(scalar_new[ii], d_lab->d_scalarINLabel, ii, patch);
  }
  new_dw->put(density_new, d_lab->d_densityINLabel, matlIndex, patch);
  new_dw->put(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch);
}

// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector 
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void 
PicardNonlinearSolver::interpolateFromFCToCC(const ProcessorGroup* ,
					     const Patch* patch,
					     DataWarehouseP& /*old_dw*/,
					     DataWarehouseP& new_dw)
{
  int matlIndex = 0;
  int nofGhostCells = 1;

  // Get the old velocity
  SFCXVariable<double> oldUVel;
  SFCYVariable<double> oldVVel;
  SFCZVariable<double> oldWVel;
  new_dw->get(oldUVel, d_lab->d_uVelocityINLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);
  new_dw->get(oldVVel, d_lab->d_vVelocityINLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);
  new_dw->get(oldWVel, d_lab->d_wVelocityINLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);

  // Get the new velocity
  SFCXVariable<double> newUVel;
  SFCYVariable<double> newVVel;
  SFCZVariable<double> newWVel;
  new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);
  new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);
  new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
	      Ghost::AroundCells, nofGhostCells);

  // Get the low and high index for the Cell Centered Variables
  IntVector idxLo = patch->getCellLowIndex();
  IntVector idxHi = patch->getCellHighIndex();

  // Allocate the interpolated velocities
  CCVariable<Vector> oldCCVel;
  CCVariable<Vector> newCCVel;
  new_dw->allocate(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
  new_dw->allocate(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch);

  // Interpolate the FC velocity to the CC
  for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
    for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
      for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	
	IntVector idx(ii,jj,kk);
	IntVector idxU(ii+1,jj,kk);
	IntVector idxV(ii,jj+1,kk);
	IntVector idxW(ii,jj,kk+1);

	// old U velocity (linear interpolation)
	double old_u = 0.5*(oldUVel[idx] + 
			    oldUVel[idxU]);
	// new U velocity (linear interpolation)
	double new_u = 0.5*(newUVel[idx] +
			    newUVel[idxU]);

	// old V velocity (linear interpolation)
	double old_v = 0.5*(oldVVel[idx] +
			    oldVVel[idxV]);
	// new V velocity (linear interpolation)
	double new_v = 0.5*(newVVel[idx] +
			    newVVel[idxV]);

	// old W velocity (linear interpolation)
	double old_w = 0.5*(oldWVel[idx] +
			    oldWVel[idxW]);
	// new W velocity (linear interpolation)
	double new_w = 0.5*(newWVel[idx] +
			    newWVel[idxW]);

	// Add the data to the CC Velocity Variables
	oldCCVel[idx] = Vector(old_u,old_v,old_w);
	newCCVel[idx] = Vector(new_u,new_v,new_w);
      }
    }
  }

  // Put the calculated stuff into the new_dw
  new_dw->put(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
  new_dw->put(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch);
}

// ****************************************************************************
// compute the residual
// ****************************************************************************
double 
PicardNonlinearSolver::computeResidual(const LevelP& /*level*/,
				       SchedulerP& /*sched*/,
				       DataWarehouseP& /*old_dw*/,
				       DataWarehouseP& /*new_dw*/)
{
  double nlresidual = 0.0;
#if 0
  SoleVariable<double> residual;
  SoleVariable<double> omg;
  // not sure of the syntax...this operation is supposed to get 
  // L1norm of the residual over the whole level
  new_dw->get(residual,"pressResidual");
  new_dw->get(omg,"pressomg");
  nlresidual = MACHINEPRECISSION + log(residual/omg);
  for (int index = 1; index <= Arches::NDIM; ++index) {
    new_dw->get(residual,"velocityResidual", index);
    new_dw->get(omg,"velocityomg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
  //for multiple scalars iterate
  for (int index = 1;index <= d_props->getNumMixVars(); index ++) {
    new_dw->get(residual,"scalarResidual", index);
    new_dw->get(omg,"scalaromg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
#endif
  return nlresidual;
}


