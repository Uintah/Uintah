//----- PicardNonlinearSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_NonlinearSolver_h
#define Uintah_Component_Arches_NonlinearSolver_h

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

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

namespace Uintah {

class NonlinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for NonlinearSolver.
      NonlinearSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for NonlinearSolver.
      virtual ~NonlinearSolver();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up of the problem specification database
      virtual void problemSetup(const ProblemSpecP& db) = 0;

      virtual void sched_interpolateFromFCToCC(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls) = 0;
      // GROUP: Schedule Action Computations :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Solve the nonlinear system, return some error code.
      //    [in] 
      //        documentation here
      //    [out] 
      //        documentation here
      virtual int nonlinearSolve( const LevelP& level,
				  SchedulerP& sched) = 0;
  
protected:
   const ProcessorGroup* d_myworld;
private:

}; // End class NonlinearSolver
} // End namespace Uintah

#endif


