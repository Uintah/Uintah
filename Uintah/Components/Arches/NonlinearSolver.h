//----- PicardNonlinearSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_NonlinearSolver_h
#define Uintah_Component_Arches_NonlinearSolver_h

/**************************************
CLASS
   NonlinearSolver
   
   Class NonlinearSolver is an abstract base class
   which defines the operations needed to implement
   a nonlinear solver for the ImplicitTimeIntegrator.

GENERAL INFORMATION
   NonlinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class NonlinearSolver is an abstract type defining the interface
   for operations necessary for solving the nonlinear systems that
   arise during an implicit integration.

WARNING
   none
****************************************/

#include <Uintah/Components/Arches/Arches.h>

namespace Uintah {
namespace ArchesSpace {

class NonlinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Blank constructor for NonlinearSolver.
      //
      NonlinearSolver();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for NonlinearSolver.
      //
      virtual ~NonlinearSolver();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Interface for Set up of the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& db) = 0;

      // GROUP: Schedule Action Computations :
      ///////////////////////////////////////////////////////////////////////
      //
      // Interface for Solve the nonlinear system, return some error code.
      //    [in] 
      //        documentation here
      //    [out] 
      //        documentation here
      //
      virtual int nonlinearSolve(double time, double deltat, 
				 const LevelP&, 
				 SchedulerP& sched,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw) = 0;
  
private:

}; // End class NonlinearSolver
}  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.8  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

