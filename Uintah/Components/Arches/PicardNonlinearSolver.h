/**************************************
CLASS
   NonlinearSolver
   
   Class PicardNonlinearSolver is a subclass of NonlinearSolver
   which implements the Fixed Point Picard iteration.[Ref Kumar's thesis]

GENERAL INFORMATION
   PicardNonlinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 1998


KEYWORDS


DESCRIPTION
   Class PicardNonlinearSolver implements the
   Fixed Point Picard iteration method is used by
   ImplicitIntegrator to solve set of nonlinear equations

WARNING
   none
****************************************/

#ifndef Uintah_Component_Arches_PicardNonlinearSolver_h
#define Uintah_Component_Arches_PicardNonlinearSolver_h

#include "Arches.h"
#include "NonlinearSolver.h"



#ifndef LACKS_NAMESPACE
using namespace UINTAH;
#endif

class PressureSolver;
class MomentumSolver;
class ScalarSolver;
class TurbulenceModel;
class Properties;
const double MACHINEPRECISSION = 14.0; //used to compute residual
class PicardNonlinearSolver:
public NonlinearSolver
{
public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  PicardNonlinearSolver();



  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for PicardNonlinearSolver.
  virtual ~PicardNonlinearSolver();


  ////////////////////////////////////////////////////////////////////////
  // Solve the nonlinear system.
  // The code returns 0 if there are no errors and
  // 1 if there is a nonlinear failure.
  //    [in] data User data needed for solve 
  virtual int nonlinearSolve(Arches* integrator);
  
  virtual void problemSetup(DatabaseP& input_db);
private:
  // Total number of nonlinear iterates
  int d_nonlinear_its;
  // nonlinear residual
  double d_residual;
    
  // Pressure Eqn Solver
  PressureSolver* d_pressSolver;
  
  // Momentum Eqn Solver 
  MomentumSolver* d_momSolver;

  // Scalar solver
  ScalarSolver* d_scalarSolver;

  // properties...solves density, temperature and specie concentrations
  Properties* d_props;
  
  // Turbulence Model
  TurbulenceModel* d_turbModel;
  
};

#endif

