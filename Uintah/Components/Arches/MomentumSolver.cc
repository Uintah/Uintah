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
// Private constructor for MomentumSolver
//****************************************************************************
MomentumSolver::MomentumSolver()
{
}

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
  // inputs/outputs for sched_buildLinearMatrix
  d_pressurePSLabel = scinew VarLabel("pressurePS",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				     SFCXVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				     SFCYVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				     SFCZVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				     CCVariable<double>::getTypeDescription() );
  d_uVelConvCoefMBLMLabel = scinew VarLabel("uVelConvCoefMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelConvCoefMBLMLabel = scinew VarLabel("vVelConvCoefMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelConvCoefMBLMLabel = scinew VarLabel("wVelConvCoefMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelCoefMBLMLabel = scinew VarLabel("uVelCoefMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefMBLMLabel = scinew VarLabel("vVelCoefMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefMBLMLabel = scinew VarLabel("wVelCoefMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				   SFCZVariable<double>::getTypeDescription() );
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
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
    //throw InvalidValue("Finite Differencing scheme "
	//	       "not supported: " + finite_diff, db);
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
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
  }
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
}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void 
MomentumSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t, int index)
{
  // compute ith componenet of velocity stencil coefficients
  // inputs : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
  // outputs: [u,v,w]VelConvCoefMBLM, [u,v,w]VelCoefMBLM
  d_discretize->calculateVelocityCoeff(pc, patch, old_dw, new_dw, 
				       delta_t, index,
				       Arches::MOMENTUM);

  // Calculate velocity source
  // inputs : [u,v,w]VelocityCPBC, densityCP, viscosityCTS
  // outputs: [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcMBLM
  d_source->calculateVelocitySource(pc, patch, old_dw, new_dw, 
				    delta_t, index,
				    Arches::MOMENTUM);

  // Velocity Boundary conditions
  //  inputs : densityCP, [u,v,w]VelocityCPBC, [u,v,w]VelCoefMBLM
  //           [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcMBLM
  //  outputs: [u,v,w]VelCoefMBLM, [u,v,w]VelLinSrcMBLM, 
  //           [u,v,w]VelNonLinSrcMBLM
  d_boundaryCondition->velocityBC(pc, patch, old_dw, new_dw, 
				  index,
				  Arches::MOMENTUM);

  // similar to mascal
  // inputs :
  // outputs:
  d_source->modifyVelMassSource(pc, patch, old_dw,
			     new_dw, delta_t, index);

  // Calculate Velocity Diagonal
  //  inputs : [u,v,w]VelCoefMBLM, [u,v,w]VelLinSrcMBLM
  //  outputs: [u,v,w]VelCoefMBLM
  d_discretize->calculateVelDiagonal(pc, patch, new_dw, new_dw, 
				     index,
				     Arches::MOMENTUM);

  // Add the pressure source terms
  // inputs :
  // outputs:
  d_source->addPressureSource(pc, patch, new_dw, new_dw, index);

}

//
// $Log$
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

