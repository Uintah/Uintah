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
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace ArchesSpace {

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

      // GROUP: Schedule Action:
      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule underrelaxation
      //
      virtual void sched_underrelax(const LevelP& level,
				       SchedulerP& sched,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule the pressure solve
      //
      virtual void sched_pressureSolve(const LevelP& level,
				       SchedulerP& sched,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) = 0;

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule the velocity solve
      //
      virtual void sched_velSolve(const LevelP& level,
				  SchedulerP& sched,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  const int index) = 0;

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
// Revision 1.6  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
  
