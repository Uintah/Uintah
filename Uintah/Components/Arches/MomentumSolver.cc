//----- MomentumSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/MomentumSolver.h>
#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Default constructor for MomentumSolver
//****************************************************************************
MomentumSolver::MomentumSolver(TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst) : 
                                   d_turbModel(turb_model), 
                                   d_boundaryCondition(bndry_cond),
				   d_physicalConsts(physConst),
				   d_generation(0)
{
  d_lab = scinew ArchesLabel();
}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
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
  if (finite_diff == "second") d_discretize = scinew Discretization();
  else 
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);

  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
   
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs") d_linearSolver = scinew RBGSSolver();
  else
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solve(const LevelP& level,
		      SchedulerP& sched,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw,
		      double time, double delta_t, int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  //DataWarehouseP matrix_dw = sched->createDataWarehouse(d_generation);
  //++d_generation;

  //computes stencil coefficients and source terms
  // require : pressurePS, [u,v,w]VelocityCPBC, densityCP, viscosityCTS
  // compute : [u,v,w]VelCoefMBLM, [u,v,w]VelConvCoefMBLM
  //           [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcMBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t, index);
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t, index);
    
  // Schedules linear velocity solve
  // require : [u,v,w]VelocityCPBC, [u,v,w]VelCoefMBLM,
  //           [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcMBLM
  // compute : [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
  //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
  //           [u,v,w]VelocityMS
  //d_linearSolver->sched_velSolve(level, sched, new_dw, matrix_dw, index);
  d_linearSolver->sched_velSolve(level, sched, new_dw, new_dw, index);
    
}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					double delta_t, int index)
{
  /*
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      // steve: requires two arguments
      // Task* tsk = scinew Task("MomentumSolver::BuildCoeff",
	// 		      patch, old_dw, new_dw, this,
	// 		      Discretization::buildLinearMatrix,
	// 		      delta_t, index);
      Task* tsk = scinew Task("MomentumSolver::BuildCoeff",
			      patch, old_dw, new_dw, this,
			      &MomentumSolver::buildLinearMatrix,
			      delta_t, index);

      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_densityCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_pressurePSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes index components of velocity
      tsk->computes(new_dw, d_uVelConvCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_vVelConvCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_wVelConvCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_uVelCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_vVelCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_wVelCoefMBLMLabel, index, patch);
      tsk->computes(new_dw, d_uVelLinSrcMBLMLabel, index, patch);
      tsk->computes(new_dw, d_vVelLinSrcMBLMLabel, index, patch);
      tsk->computes(new_dw, d_wVelLinSrcMBLMLabel, index, patch);
      tsk->computes(new_dw, d_uVelNonLinSrcMBLMLabel, index, patch);
      tsk->computes(new_dw, d_vVelNonLinSrcMBLMLabel, index, patch);
      tsk->computes(new_dw, d_wVelNonLinSrcMBLMLabel, index, patch);

      sched->addTask(tsk);
    }
  }
  */

  // Create tasks for each of the actions in BuildLinearMatrix for the
  // three Momentum Solves
  int eqnType = Arches::MOMENTUM;
  std::string dir("XMOM");
  if (index == 0) dir = "XMOM";
  else if (index == 1) dir = "YMOM";
  else if (index == 2) dir = "ZMOM";
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // Task 1 : CalculateVelocityCoeff
    // requires : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
    // computes : [u,v,w]VelConvCoefMBLM, [u,v,w]VelCoefM0
    {
      Task* tsk = scinew Task("Momentum::VelCoef"+dir,
			      patch, old_dw, new_dw, d_discretize,
			      &Discretization::calculateVelocityCoeff, 
			      delta_t, eqnType, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->densityCPLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->viscosityCTSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      
      tsk->computes(new_dw, d_lab->uVelCoefM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelCoefM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelCoefM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->uVelConvCoefMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelConvCoefMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelConvCoefMBLMLabel[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }
    
    // Task 2: CalculateVelocitySource
    // requires : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
    // computes : [u,v,w]VelLinSrcM0, [u,v,w]VelNonLinSrcM0
    {
      Task* tsk = scinew Task("Momentum::VelSource"+dir,
			      patch, old_dw, new_dw, d_source,
			      &Source::calculateVelocitySource, 
			      delta_t, eqnType, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->densityCPLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->viscosityCTSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      
      tsk->computes(new_dw, d_lab->uVelLinSrcM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelLinSrcM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelLinSrcM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->uVelNonLinSrcM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelNonLinSrcM0Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelNonLinSrcM0Label[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }

    // Task 3: VelocityBC
    // requires : densityCP, [u,v,w]VelocityCPBC, [u,v,w]VelCoefM0
    //            [u,v,w]VelLinSrcM0, [u,v,w]VelNonLinSrcM0
    // computes : [u,v,w]VelCoefM1, [u,v,w]VelLinSrcM1, 
    //            [u,v,w]VelNonLinSrcM1
    {
      Task* tsk = scinew Task("Momentum::VelBC"+dir,
			      patch, old_dw, new_dw, d_boundaryCondition,
			      &BoundaryCondition::velocityBC, 
			      eqnType, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelCoefM0Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelCoefM0Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelCoefM0Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      
      tsk->computes(new_dw, d_lab->uVelCoefM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelCoefM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelCoefM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->uVelLinSrcM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelLinSrcM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelLinSrcM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->uVelNonLinSrcM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelNonLinSrcM1Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelNonLinSrcM1Label[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }

    // Task 4: Modify Velocity Mass Source
    // requires : [u,v,w]VelocityCPBC, [u,v,w]VelCoefM1, 
    //            [u,v,w]VelConvCoefMBLM, [u,v,w]VelLinSrcM1, 
    //            [u,v,w]VelNonLinSrcM1
    // computes : [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcM2
    {
      Task* tsk = scinew Task("Momentum::VelMassSource"+dir,
			      patch, old_dw, new_dw, d_source,
			      &Source::modifyVelMassSource, 
			      delta_t, eqnType, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelConvCoefMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelConvCoefMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelConvCoefMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelNonLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelNonLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelNonLinSrcM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);

      tsk->computes(new_dw, d_lab->uVelLinSrcMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelLinSrcMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelLinSrcMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->uVelNonLinSrcM2Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelNonLinSrcM2Label[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelNonLinSrcM2Label[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }

    // Task 5: Calculate Velocity Diagonal
    // requires : [u,v,w]VelCoefM1, [u,v,w]VelLinSrcMBLM
    // computes : [u,v,w]VelCoefMBLM
    {
      Task* tsk = scinew Task("Momentum::VelDiagonal"+dir,
			      patch, old_dw, new_dw, d_discretize,
			      &Discretization::calculateVelDiagonal, 
			      eqnType, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelCoefM1Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelLinSrcMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelLinSrcMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelLinSrcMBLMLabel[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);

      tsk->computes(new_dw, d_lab->uVelCoefMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelCoefMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelCoefMBLMLabel[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }

    // Task 6: Add the pressure source terms
    // requires : [u,v,w]VelNonlinSrcM2, pressurePS, densityCP(old_dw), 
    //            [u,v,w]VelocityCPBC
    // computes : [u,v,w]VelNonlinSrcMBLM
    {
      Task* tsk = scinew Task("Momentum::AddPresSrc"+dir,
			      patch, old_dw, new_dw, d_source,
			      &Source::addPressureSource, 
			      delta_t, index);
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->densityCPLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->pressurePSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->uVelNonLinSrcM2Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->vVelNonLinSrcM2Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->wVelNonLinSrcM2Label[index], matlIndex, patch, 
		    Ghost::None, numGhostCells);
      
      tsk->computes(new_dw, d_lab->uVelNonLinSrcMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->vVelNonLinSrcMBLMLabel[index], matlIndex, patch);
      tsk->computes(new_dw, d_lab->wVelNonLinSrcMBLMLabel[index], matlIndex, patch);
      
      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
/*
void 
MomentumSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t, int index)
{
  // compute ith componenet of velocity stencil coefficients
  // inputs : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
  // outputs: [u,v,w]VelConvCoefMBLM, [u,v,w]VelCoefM0
  d_discretize->calculateVelocityCoeff(pc, patch, old_dw, new_dw, 
				       delta_t, 
				       Arches::MOMENTUM);

  // Calculate velocity source
  // inputs : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
  // outputs: [u,v,w]VelLinSrcM0, [u,v,w]VelNonLinSrcM0
  d_source->calculateVelocitySource(pc, patch, old_dw, new_dw, 
				    delta_t, 
				    Arches::MOMENTUM);

  // Velocity Boundary conditions
  //  inputs : densityCP, [u,v,w]VelocityCPBC, [u,v,w]VelCoefM0
  //           [u,v,w]VelLinSrcM0, [u,v,w]VelNonLinSrcM0
  //  outputs: [u,v,w]VelCoefM1, [u,v,w]VelLinSrcM1, 
  //           [u,v,w]VelNonLinSrcM1
  d_boundaryCondition->velocityBC(pc, patch, old_dw, new_dw, 
				  Arches::MOMENTUM);

  // Modify Velocity Mass Source
  //  inputs : [u,v,w]VelocityCPBC, [u,v,w]VelCoefM1, 
  //           [u,v,w]VelConvCoefMBLM, [u,v,w]VelLinSrcM1, 
  //           [u,v,w]VelNonLinSrcM1
  //  outputs: [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcM2
  d_source->modifyVelMassSource(pc, patch, old_dw,
				new_dw, delta_t, 
				Arches::MOMENTUM);

  // Calculate Velocity Diagonal
  //  inputs : [u,v,w]VelCoefM1, [u,v,w]VelLinSrcMBLM
  //  outputs: [u,v,w]VelCoefMBLM
  d_discretize->calculateVelDiagonal(pc, patch, new_dw, new_dw, 
				     Arches::MOMENTUM);

  // Add the pressure source terms
  // inputs :[u,v,w]VelNonlinSrcM2, pressurePS, densityCP(old_dw), 
  // [u,v,w]VelocityCPBC
  // outputs:[u,v,w]VelNonlinSrcMBLM
  d_source->addPressureSource(pc, patch, new_dw, new_dw, delta_t, index);

}
*/

//
// $Log$
// Revision 1.19  2000/07/19 06:30:01  bbanerje
// ** MAJOR CHANGES **
// If you want to get the old code go two checkins back.
//
// Revision 1.18  2000/07/18 22:33:51  bbanerje
// Changes to PressureSolver for put error. Added ArchesLabel.
//
// Revision 1.17  2000/07/17 22:06:58  rawat
// modified momentum source
//
// Revision 1.16  2000/07/12 07:35:46  bbanerje
// Added stuff for mascal : Rawat: Labels and dataWarehouse in velsrc need to be corrected.
//
// Revision 1.15  2000/07/08 23:42:54  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.14  2000/07/03 05:30:14  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.13  2000/06/29 22:56:42  bbanerje
// Changed FCVars to SFC[X,Y,Z]Vars, and added the neceesary getIndex calls.
//
// Revision 1.12  2000/06/22 23:06:34  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.11  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.10  2000/06/21 06:00:12  bbanerje
// Added two labels that were causing seg violation.
//
// Revision 1.9  2000/06/18 01:20:15  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.8  2000/06/17 07:06:24  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.6  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.5  2000/06/07 06:13:54  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.4  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

