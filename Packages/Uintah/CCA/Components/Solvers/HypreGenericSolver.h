/*--------------------------------------------------------------------------
CLASS
   HypreGenericSolver
   
   A generic Hypre solver driver.

GENERAL INFORMATION

   File: HypreGenericSolver.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverParams.

DESCRIPTION
   Class HypreGenericSolver is a base class for Hypre solvers. It uses the
   generic HypreDriver and fetches only the data it can work with (either
   Struct, SStruct, or 

   preconditioners. It does not know about the internal Hypre interfaces
   like Struct and SStruct. Instead, it uses the generic HypreDriver
   and HypreGenericSolver::newSolver to determine the specific Hypre
   interface and solver, based on the parameters in HypreSolverParams.
   The solver is called through the solve() function. This is also the
   task-scheduled function in HypreSolverAMR::scheduleSolve() that is
   activated by Components/ICE/impICE.cc.
  
WARNING
   solve() is a generic function for all Types, but this may need to change
   in the future. Currently only CC is implemented for the pressure solver
   in implicit [AMR] ICE.
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>

namespace Uintah {

  class HypreGenericSolver {

    /*========================== PUBLIC SECTION ==========================*/
  public:
  
    /*---------- Types ----------*/

    enum SolverType {
      SMG, PFMG, SparseMSG, CG, Hybrid, GMRES, AMG, FAC
    };

    enum PrecondType {
      PrecondNA, // No preconditioner, use solver directly
      PrecondSMG, PrecondPFMG, PrecondSparseMSG, PrecondJacobi,
      PrecondDiagonal, PrecondAMG, PrecondFAC
    };

    /* Solver results are output to this struct */
    struct Results {
      int        numIterations;   // Number of solver iterations performed
      double     finalResNorm;    // Final residual norm ||A*x-b||_2
    };
    
    HypreGenericSolver(const SolverType& solverType,
                       HypreDriver::HypreInterface& hypreInterface)
      : _solverType(solverType), _hypreInterface(hypreInterface)
      {
        _results.numIterations = 0;
        _results.finalResNorm  = 1.23456e+30; // Large number
      }
  
    virtual ~HypreGenericSolver(void) {
      // TODO: move to HypreDriver
      /* Destroy graph objects */
      //      HYPRE_SStructGraphDestroy(_graph);

      /* Destroy matrix, RHS, solution objects */
      //      HYPRE_SStructMatrixDestroy(_A);
      //      HYPRE_SStructVectorDestroy(_b);
      //      HYPRE_SStructVectorDestroy(_x);
    }

    virtual void setup(void);
    virtual void solve(void);

    /* Utilities */
    virtual void printMatrix(const string& fileName = "output");
    virtual void printRHS(const string& fileName = "output_b");
    virtual void printSolution(const string& fileName = "output_x");

    /*========================== PROTECTED SECTION ==========================*/
  protected:

    /* Utilities */
    static SolverType solverFromTitle(const string& solverTitle);
    static PrecondType precondFromTitle(const string& precondTitle);
    static HypreDriver::InterfaceType solverInterface
      (const SolverType& solverType);

    /*========================== PRIVATE SECTION ==========================*/
  private:
    /*---------- Data members ----------*/
    SolverType            _solverType;
    HypreDriver::HypreInterface& _hypreInterface;
    bool                  _requiresPar;   // Does solver require Par input?
    int                   _solverID;      // Hypre solver ID
    Results               _results;       // Solver results are stored to here

    void makeGraph();
  };
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h
