//----- PressureSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
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
PressureSolver::PressureSolver()
{
}

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
PressureSolver::PressureSolver(int nDim,
			       TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst): 
                                     d_NDIM(nDim),
                                     d_turbModel(turb_model), 
                                     d_boundaryCondition(bndry_cond),
				     d_physicalConsts(physConst),
				     d_generation(0)
{
  // Inputs
  d_pressureINLabel = scinew VarLabel("pressureIN",
			     CCVariable<double>::getTypeDescription() );
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				 FCVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				 FCVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				 FCVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
			       CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
			       CCVariable<double>::getTypeDescription() );

  // Computed
  d_uVelConvCoefPBLMLabel = scinew VarLabel("uVelConvectCoefPBLM",
				   FCVariable<double>::getTypeDescription() );
  d_vVelConvCoefPBLMLabel = scinew VarLabel("vVelConvectCoefPBLM",
				   FCVariable<double>::getTypeDescription() );
  d_wVelConvCoefPBLMLabel = scinew VarLabel("wVelConvectCoefPBLM",
				   FCVariable<double>::getTypeDescription() );
  d_uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
			       FCVariable<double>::getTypeDescription() );
  d_vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
			       FCVariable<double>::getTypeDescription() );
  d_wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
			       FCVariable<double>::getTypeDescription() );
  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				 FCVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				 FCVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				 FCVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    FCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    FCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    FCVariable<double>::getTypeDescription() );
  d_presCoefPBLMLabel = scinew VarLabel("presCoefPBLM",
			       CCVariable<double>::getTypeDescription() );
  d_presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM",
				 CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
PressureSolver::~PressureSolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PressureSolver");
  db->require("ref_point", d_pressRef);
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
    throw InvalidValue("Linear solver option"
		       " not supported" + linear_sol);
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
  }
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule solve of linearized pressure equation
//****************************************************************************
void PressureSolver::solve(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw,
			   double time, double delta_t)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  //DataWarehouseP matrix_dw = sched->createDataWarehouse(d_generation);
  //++d_generation;

  //computes stencil coefficients and source terms
  // require : pressureIN, densitySIVBC, viscosityCTS, [u,v,w]VelocitySIVBC
  // compute : uVelConvCoefPBLM, vVelConvCoefPBLM, wVelConvCoefPBLM
  //           uVelCoefPBLM, vVelCoefPBLM, wVelCoefPBLM, uVelLinSrcPBLM
  //           vVelLinSrcPBLM, wVelLinSrcPBLM, uVelNonLinSrcPBLM 
  //           vVelNonLinSrcPBLM, wVelNonLinSrcPBLM, presCoefPBLM 
  //           presLinSrcPBLM, presNonLinSrcPBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t);

  //residual at the start of linear solve
  // this can be part of linear solver
#if 0
  calculateResidual(level, sched, new_dw, matrix_dw);
  calculateOrderMagnitude(level, sched, new_dw, matrix_dw);
#endif

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);
  d_linearSolver->sched_pressureSolve(level, sched, old_dw, new_dw);

  // Schedule Calculation of pressure norm
  // require :
  // compute :
  //sched_normPressure(level, sched, new_dw, matrix_dw);
  sched_normPressure(level, sched, old_dw, new_dw);
  
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					double delta_t)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::BuildCoeff",
			   patch, old_dw, new_dw, this,
			   &PressureSolver::buildLinearMatrix, delta_t);

      int numGhostCells = 0;
      int matlIndex = 0;

      // Requires
      tsk->requires(old_dw, d_pressureINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      tsk->computes(new_dw, d_uVelConvCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelConvCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelConvCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presCoefPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presNonLinSrcPBLMLabel, matlIndex, patch);
     
      sched->addTask(tsk);
    }

  }
}

//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t)
{
  // compute all three componenets of velocity stencil coefficients
  for(int index = 1; index <= Arches::NDIM; ++index) {

    // Calculate Velocity Coeffs :
    //  inputs : [u,v,w]VelocitySIVBC, densitySIVBC, viscosityCTS
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 
    d_discretize->calculateVelocityCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, index,
					 Discretization::PRESSURE);

    // Calculate Velocity source
    //  inputs : [u,v,w]VelocitySIVBC, densitySIVBC, viscosityCTS
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    d_source->calculateVelocitySource(pc, patch, new_dw, new_dw, 
				      delta_t, index,
				      Discretization::PRESSURE);

    // Calculate the Velocity BCS
    //  inputs : densitySIVBC, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    d_boundaryCondition->velocityBC(pc, patch, new_dw, new_dw, 
				    index,
				    Discretization::PRESSURE);

    // Modify Velocity Mass Source
    //  inputs :
    //  outputs:
    d_source->modifyVelMassSource(pc, patch, new_dw, new_dw, delta_t, index);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM
    d_discretize->calculateVelDiagonal(pc, patch, new_dw, new_dw, 
				       index,
				       Discretization::PRESSURE);
  }

  // Calculate Pressure Coeffs
  //  inputs : pressureIN, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
  //  outputs: presCoefPBLM 
  d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, delta_t);

  // Calculate Pressure Source
  //  inputs : pressureIN, [u,v,w]VelocitySIVBC, densitySIVBC,
  //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
  //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
  d_source->calculatePressureSource(pc, patch, old_dw, new_dw, delta_t);

  // Calculate Pressure BC
  //  inputs : pressureIN, presCoefPBLM
  //  outputs: presCoefPBLM
  d_boundaryCondition->pressureBC(pc, patch, old_dw, new_dw);

  // Calculate Pressure Diagonal
  //  inputs : presCoefPBLM, presLinSrcPBLM
  //  outputs: presCoefPBLM 
  d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw);

}

//****************************************************************************
// Schedule the creation of the .. more documentation here
//****************************************************************************
void 
PressureSolver::sched_normPressure(const LevelP& ,
		                   SchedulerP& ,
		                   DataWarehouseP& ,
		                   DataWarehouseP& )
{
}  

//****************************************************************************
// Actually do normPressure
//****************************************************************************
void 
PressureSolver::normPressure(const Patch* ,
	                     SchedulerP& ,
	                     const DataWarehouseP& ,
	                     DataWarehouseP& )
{
}

//
// $Log$
// Revision 1.25  2000/06/22 23:06:35  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.24  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.23  2000/06/21 06:49:21  bbanerje
// Straightened out some of the problems in data location .. still lots to go.
//
// Revision 1.22  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.21  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.20  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.19  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.18  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.17  2000/06/07 06:13:55  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.15  2000/06/04 23:57:46  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
// Revision 1.14  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
