//
// $Id$
//

#ifndef Uintah_Components_Arches_LinearSolver_h
#define Uintah_Components_Arches_LinearSolver_h

/**************************************
CLASS
   LinearSolver
   
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   LinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.



WARNING
none
****************************************/

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
class ProcessorGroup;
class LoadBalancer;
namespace ArchesSpace {
class ArchesVariables;
class ArchesLabel;
class CellInformation;
// class StencilMatrix;
using namespace SCICore::Containers;

class LinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a LinearSolver.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //
      LinearSolver();


      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual Destructor
      //
      virtual ~LinearSolver();


      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      //
      // Setup the problem (etc.)
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;


      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Underrelaxation
      //
      virtual void computePressUnderrelax(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, ArchesVariables* vars)= 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Solve
      //
      virtual void pressLisolve(const ProcessorGroup* pc,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw, ArchesVariables* vars, 
				const ArchesLabel* lab) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate pressure residuals
      //
      virtual void computePressResidual(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for pressure equation
      //
      virtual void computePressOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Underrelaxation
      //
      virtual void computeVelUnderrelax(const ProcessorGroup* pc,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw, int index,
					ArchesVariables* vars)= 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Velocity Solve
      //
      virtual void velocityLisolve(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw, int index,
				   double delta_t,
				   ArchesVariables* vars,
				   CellInformation* cellinfo,
				   const ArchesLabel* lab) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate Velocity residuals
      //
      virtual void computeVelResidual(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, int index,
				      ArchesVariables* vars) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for Velocity equation
      //
      virtual void computeVelOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Underrelaxation
      //
      virtual void computeScalarUnderrelax(const ProcessorGroup* pc,
					   const Patch* patch,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw, int index,
					   ArchesVariables* vars)= 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Scalar Solve
      //
      virtual void scalarLisolve(const ProcessorGroup* pc,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw, int index,
				 double delta_t,
				 ArchesVariables* vars,
				 CellInformation* cellinfo,
				 const ArchesLabel* lab) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate Scalar residuals
      //
      virtual void computeScalarResidual(const ProcessorGroup* pc,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw, int index,
					 ArchesVariables* vars) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate order of magnitude term for scalar equation
      //
      virtual void computeScalarOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


   virtual void matrixCreate(const LevelP& level, LoadBalancer* lb) = 0;
   virtual void setPressMatrix(const ProcessorGroup* pc, const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw, 
			       ArchesVariables* vars,
			       const ArchesLabel* lab,
			       int patchNumber) = 0;
   

   virtual void pressLinearSolve() = 0;
   virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars) = 0;
   virtual void destroyMatrix() = 0;

protected:

private:

}; // End class LinearSolve

} // End namespace ArchesSpace
} // End namespace Uintah
#endif  

//
// $Log$
// Revision 1.15  2000/09/21 21:45:05  rawat
// added petsc parallel stuff
//
// Revision 1.14  2000/09/20 18:05:33  sparker
// Adding support for Petsc and per-processor tasks
//
// Revision 1.13  2000/08/15 00:23:32  rawat
// added explicit solve for momentum and scalar eqns
//
// Revision 1.12  2000/08/12 23:53:18  bbanerje
// Added Linegs part to the solver.
//
// Revision 1.11  2000/08/11 21:26:36  rawat
// added linear solver for pressure eqn
//
// Revision 1.10  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.9  2000/08/01 06:18:37  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.8  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.7  2000/06/17 07:06:24  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.6  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
  
