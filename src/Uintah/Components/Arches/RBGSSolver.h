//
// $Id$
//

#ifndef Uintah_Components_Arches_RBGSSolver_h
#define Uintah_Components_Arches_RBGSSolver_h

/**************************************
CLASS
   RBGSSolver
   
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver

GENERAL INFORMATION
   RBGSSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver




WARNING
none
****************************************/

#include <Uintah/Components/Arches/LinearSolver.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
class ProcessorGroup;
namespace ArchesSpace {
  //class LinearSolver;
using namespace SCICore::Containers;

class RBGSSolver: public LinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a RBGSSolver.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //
      // Default constructor.
      //
      RBGSSolver();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual Destructor
      //
      virtual ~RBGSSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      //
      // Problem setup
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP:  Schedule Action :
      ////////////////////////////////////////////////////////////////////////
      //
      // Underrelaxation
      //
      void sched_underrelax(const LevelP& level,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Solve
      //
      void sched_pressureSolve(const LevelP& level,
			       SchedulerP& sched,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Solve
      //
      void sched_velSolve(const LevelP& level,
			  SchedulerP& sched,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
			  int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Solve
      //
      void sched_scalarSolve(const LevelP& level,
			     SchedulerP& sched,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw,
			     int index);
protected:

private:

      // GROUP:  Actual Action :
      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Underrelaxation
      //
      void press_underrelax(const ProcessorGroup* pc,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Solve
      //
      void press_lisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate pressure residuals
      //
      void press_residCalculation(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Underrelaxation
      //
      void vel_underrelax(const ProcessorGroup* pc,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw, 
			  int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Solve
      //
      void vel_lisolve(const ProcessorGroup* pc,
		       const Patch* patch,
		       DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw, 
		       int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate Velocity residuals
      //
      void vel_residCalculation(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, 
				int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Underrelaxation
      //
      void scalar_underrelax(const ProcessorGroup* pc,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw, 
			     int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Solve
      //
      void scalar_lisolve(const ProcessorGroup* pc,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw, 
			  int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate Scalar residuals
      //
      void scalar_residCalculation(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw, 
				   int index);

      int d_maxSweeps;
      double d_convgTol; // convergence tolerence
      double d_underrelax;
      double d_initResid;
      double d_residual;

      // const VarLabel *
      // inputs (Pressure Solve)
      const VarLabel* d_pressureINLabel;
      const VarLabel* d_presCoefPBLMLabel;
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // computes (Pressure Solve)
      const VarLabel* d_presResidualPSLabel;
      const VarLabel* d_presCoefPSLabel;
      const VarLabel* d_presNonLinSrcPSLabel;
      const VarLabel* d_pressurePSLabel;

      // inputs (Momentum Solve)
      const VarLabel* d_uVelocityCPBCLabel;
      const VarLabel* d_uVelCoefMBLMLabel;
      const VarLabel* d_uVelNonLinSrcMBLMLabel;
      const VarLabel* d_vVelocityCPBCLabel;
      const VarLabel* d_vVelCoefMBLMLabel;
      const VarLabel* d_vVelNonLinSrcMBLMLabel;
      const VarLabel* d_wVelocityCPBCLabel;
      const VarLabel* d_wVelCoefMBLMLabel;
      const VarLabel* d_wVelNonLinSrcMBLMLabel;

      // computes (Momentum Solve)
      const VarLabel* d_uVelResidualMSLabel;
      const VarLabel* d_uVelCoefMSLabel;
      const VarLabel* d_uVelNonLinSrcMSLabel;
      const VarLabel* d_uVelocityMSLabel;
      const VarLabel* d_vVelResidualMSLabel;
      const VarLabel* d_vVelCoefMSLabel;
      const VarLabel* d_vVelNonLinSrcMSLabel;
      const VarLabel* d_vVelocityMSLabel;
      const VarLabel* d_wVelResidualMSLabel;
      const VarLabel* d_wVelCoefMSLabel;
      const VarLabel* d_wVelNonLinSrcMSLabel;
      const VarLabel* d_wVelocityMSLabel;

      // inputs (Scalar Solve)
      const VarLabel* d_scalarSPLabel;
      const VarLabel* d_scalCoefSBLMLabel;
      const VarLabel* d_scalNonLinSrcSBLMLabel;

      // computes (Scalar Solve)
      const VarLabel* d_scalResidualSSLabel;
      const VarLabel* d_scalCoefSSLabel;
      const VarLabel* d_scalNonLinSrcSSLabel;
      const VarLabel* d_scalarSSLabel;

}; // End class RBGSSolver.h

} // End namespace ArchesSpace
} // End namespace Uintah

#endif  
  
//
// $Log$
// Revision 1.11  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.10  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.9  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.8  2000/06/12 21:29:59  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.7  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.6  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
