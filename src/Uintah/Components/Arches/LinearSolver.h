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
namespace ArchesSpace {
class ArchesVariables;
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
			DataWarehouseP& new_dw, ArchesVariables* vars) = 0;

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
      // Pressure Underrelaxation
      //
      virtual void computeVelUnderrelax(const ProcessorGroup* pc,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw, int index,
					ArchesVariables* vars)= 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Pressure Solve
      //
      virtual void velocityLisolve(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw, int index,
				   ArchesVariables* vars) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Calculate pressure residuals
      //
      virtual void computeVelResidual(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, int index,
				      ArchesVariables* vars) = 0;



      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule the scalar solve
      //
      virtual void sched_scalarSolve(const LevelP& level,
				     SchedulerP& sched,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     const int index) = 0;
protected:

private:

}; // End class LinearSolve

} // End namespace ArchesSpace
} // End namespace Uintah
#endif  

//
// $Log$
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
  
