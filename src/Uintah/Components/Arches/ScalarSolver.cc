//----- ScalarSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/ScalarSolver.h>
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
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Private constructor for PressureSolver
//****************************************************************************
ScalarSolver::ScalarSolver()
{
}

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
ScalarSolver::ScalarSolver(TurbulenceModel* turb_model,
			   BoundaryCondition* bndry_cond,
			   PhysicalConstants* physConst) : 
                                 d_turbModel(turb_model), 
                                 d_boundaryCondition(bndry_cond),
				 d_physicalConsts(physConst),
				 d_generation(0)
{
  d_scalarSPLabel = scinew VarLabel("scalarSP",
				CCVariable<double>::getTypeDescription() );
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				FCVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				FCVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				FCVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
				CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				CCVariable<double>::getTypeDescription() );
  d_scalCoefSBLMLabel = scinew VarLabel("scalCoefSBLM",
				CCVariable<double>::getTypeDescription() );
  d_scalLinSrcSBLMLabel = scinew VarLabel("scalLinSrcSBLM",
				CCVariable<double>::getTypeDescription() );
  d_scalNonLinSrcSBLMLabel = scinew VarLabel("scalNonLinSrcSBLM",
				CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
ScalarSolver::~ScalarSolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MixtureFractionSolver");
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
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ScalarSolver::solve(const LevelP& level,
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
  // requires : scalarSP, [u,v,w]VelocityMS, densitySIVBC, viscosityCTS
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t, index);
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t, index);
    
  // Schedule the scalar solve
  // require : scalarSP, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSS
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  d_linearSolver->sched_scalarSolve(level, sched, old_dw, new_dw, index);
    
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(const LevelP& level,
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
      //Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
	//		      patch, old_dw, new_dw, this,
	//		      Discretization::buildLinearMatrix,
	//		      delta_t, index);
      Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
			      patch, old_dw, new_dw, this,
			      &ScalarSolver::buildLinearMatrix,
			      delta_t, index);

      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires
      tsk->requires(old_dw, d_scalarSPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      // added one more argument of index to specify scalar component
      tsk->computes(new_dw, d_scalCoefSBLMLabel, index, patch);
      tsk->computes(new_dw, d_scalLinSrcSBLMLabel, index, patch);
      tsk->computes(new_dw, d_scalNonLinSrcSBLMLabel, index, patch);

      sched->addTask(tsk);
    }

  }
}

//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t, int index)
{
  // compute ith componenet of velocity stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densitySIVBC, viscosityCTS
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarCoeff(pc, patch, old_dw, new_dw, 
				     delta_t, index);

  // Calculate scalar source terms
  // inputs : [u,v,w]VelocityMS, scalarSP, densitySIVBC, viscosityCTS
  // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
  d_source->calculateScalarSource(pc, patch, old_dw, new_dw, 
				  delta_t, index);

  // Calculate the scalar boundary conditions
  // inputs : scalarSP, scalCoefSBLM
  // outputs: scalCoefSBLM
  d_boundaryCondition->scalarBC(pc, patch, old_dw, new_dw, index);

  // similar to mascal
  // inputs :
  // outputs:
  d_source->modifyScalarMassSource(pc, patch, new_dw,
				   new_dw, delta_t, index);

  // Calculate the scalar diagonal terms
  // inputs : scalCoefSBLM, scalLinSrcSBLM
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarDiagonal(pc, patch, new_dw,
				     new_dw, index);
}

//
// $Log$
// Revision 1.13  2000/06/28 08:14:53  bbanerje
// Changed the init routines a bit.
//
// Revision 1.12  2000/06/22 23:06:37  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.11  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.10  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.9  2000/06/18 01:20:17  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.8  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.6  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.5  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.4  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.3  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

