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
#include <Uintah/Components/Arches/ArchesVariables.h>

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

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Underrelaxation
      //
      void computePressUnderrelax(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Solve
      //
      void pressLisolve(const ProcessorGroup* pc,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate pressure residuals
      //
      void computePressResidual(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);


      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for pressure equation
      //
      void computePressOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);


      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Underrelaxation
      //
      void computeVelUnderrelax(const ProcessorGroup* pc,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw, int index,
					ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Solve
      //
      void velocityLisolve(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw, int index,
				   ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate Velocity residuals
      //
      void computeVelResidual(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, int index,
				      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for Velocity equation
      //
      void computeVelOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate scalar residuals
      //
      void computeScalarResidual(const ProcessorGroup* pc,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw, 
				 int index,
				 ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for scalar equation
      //
      void computeScalarOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Underrelaxation
      //
      void computeScalarUnderrelax(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw, 
				   int index,
				   ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Solve
      //
      void scalarLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw, 
			 int index,
			 ArchesVariables* vars);

protected:

private:

      int d_maxSweeps;
      double d_convgTol; // convergence tolerence
      double d_underrelax;
      double d_initResid;
      double d_residual;

}; // End class RBGSSolver.h

} // End namespace ArchesSpace
} // End namespace Uintah

#endif  
  
//
// $Log$
// Revision 1.15  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.14  2000/08/01 06:18:38  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.13  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.12  2000/07/08 08:03:34  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
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
