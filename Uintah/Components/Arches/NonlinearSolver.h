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

#ifndef Uintah_Component_Arches_NonlinearSolver_h
#define Uintah_Component_Arches_NonlinearSolver_h

#include "Arches.h"

#ifndef LACKS_NAMESPACE
using namespace UINTAH;
#endif


class NonlinearSolver
{
public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Blank constructor for NonlinearSolver.
  NonlinearSolver();

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for NonlinearSolver.
  virtual ~NonlinearSolver();


  ////////////////////////////////////////////////////////////////////////
  // Solve the nonlinear system, return some error code.
  //    [in] data User data needed for solve 
  virtual int nonlinearSolve(Arches* integrator) = 0;
  
  virtual void problemSetup(DatabaseP& db) = 0;

private:

};

#endif

