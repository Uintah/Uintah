/*--------------------------------------------------------------------------
CLASS
   HypreDriver
   
   Wrapper of a Hypre solver for a particular variable type.

GENERAL INFORMATION

   File: HypreDriver.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HYPRE_Struct, HYPRE_SStruct, HYPRE_ParCSR,
   HypreGenericSolver, HypreSolverParams, RefCounted, solve, HypreSolverAMR.

DESCRIPTION 
   Class HypreDriver is a wrapper for calling Hypre solvers
   and preconditioners. It allows access to multiple Hypre interfaces:
   Struct (structured grid), SStruct (composite AMR grid), ParCSR
   (parallel compressed sparse row representation of the matrix).
   Only one of them is activated per solver, by running
   makeLinearSystem(), thereby creating the objects (usually A,b,x) of
   the specific interface.  The solver can then be constructed from
   the interface data. If required by the solver, HypreDriver converts
   the data from one Hypre interface type to another.  HypreDriver is
   also responsible for deleting all Hypre objects.
   HypreGenericSolver::newSolver determines the specific Hypre
   interface and solver, based on the parameters in HypreSolverParams.
   HypreDriver::solve() is the task-scheduled function in
   HypreSolverAMR::scheduleSolve() that is activated by
   Components/ICE/impICE.cc.

WARNING
   * solve() is a generic function for all Types, but this *might* need
   to change in the future. Currently only CC is implemented for the
   pressure solver in implicit [AMR] ICE.
   * If we intend to use other Hypre system interfaces (e.g., IJ interface),
   their data types (Matrix, Vector, etc.) should be added to the data
   members of this class. Specific solvers deriving from HypreSolverGeneric
   should have a specific Hypre interface they work with that exists in
   HypreDriver.
   * Each new interface requires its own makeLinearSystem() -- construction
   of the linear system objects, and getSolution() -- getting the solution
   vector back to Uintah.
   * This interface is written for Hypre 1.9.0b (released 2005). However,
   it may still work with the Struct solvers in earlier Hypre versions (e.g., 
   1.7.7).
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriver_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

// hypre includes
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_sstruct_ls.h>
#include <krylov.h>

namespace Uintah {

  // Forward declarations
  class HypreSolverParams;

  //---------- Types ----------
  
  enum HypreInterface {       // Hypre system interface for the solver
    HypreStruct  = 0x1,
    HypreSStruct = 0x2,
    HypreParCSR  = 0x4
  };
  
  enum CoarseFineViewpoint {
    DoingCoarseToFine,
    DoingFineToCoarse
  };
  
  enum ConstructionStatus {
    DoingGraph,
    DoingMatrix
  };
  
  enum BoxSide {  // Left/right boundary in each dim
    LeftSide  = -1,
    RightSide = 1,
    NASide    = 3
  };
  
  template<class Types>
  class HypreDriver : public RefCounted {

    //========================== PUBLIC SECTION ==========================
  public:

    //---------- Construction & Destruction ----------

    HypreDriver(const Level* level,
                const MaterialSet* matlset,
                const VarLabel* A, Task::WhichDW which_A_dw,
                const VarLabel* x, bool modifies_x,
                const VarLabel* b, Task::WhichDW which_b_dw,
                const VarLabel* guess,
                Task::WhichDW which_guess_dw,
                const HypreSolverParams* params) :
      level(level), matlset(matlset),
      A_label(A), which_A_dw(which_A_dw),
      X_label(x), modifies_x(modifies_x),
      B_label(b), which_b_dw(which_b_dw),
      guess_label(guess), which_guess_dw(which_guess_dw),
      params(params), _activeInterface(0)
      {}    
    virtual ~HypreDriver(void);

    // Main solve function
    void solve(const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               Handle<HypreDriver >);

    // Set up linear system, read back solution
    virtual void makeLinearSystem(const int matl) = 0;
    virtual void getSolution(const int matl) = 0;

    // Set up & destroy preconditioners
    virtual void setupPrecond(void) = 0;
    virtual void destroyPrecond(void) = 0;

    // Printouts
    virtual void printMatrix(const string& fileName = "output") = 0;
    virtual void printRHS(const string& fileName = "output_b") = 0;
    virtual void printSolution(const string& fileName = "output_x") = 0;

    //========================== PROTECTED SECTION ==========================
  protected:

    virtual void initialize(void) = 0;
    virtual void clear(void) = 0;

    // Utilities
    void printValues(const Patch* patch,
                     const int stencilSize,
                     const int numCells,
                     const double* values = 0,
                     const double* rhsValues = 0,
                     const double* solutionValues = 0);

    //---------- Data members ----------
    // Uintah input data
    const Level*             level;
    const MaterialSet*       matlset;
    const VarLabel*          A_label;
    Task::WhichDW            which_A_dw;
    const VarLabel*          X_label;
    bool                     modifies_x;
    const VarLabel*          B_label;
    Task::WhichDW            which_b_dw;
    const VarLabel*          guess_label;
    Task::WhichDW            which_guess_dw;
    const HypreSolverParams* params;
    
    // Assigned inside solve() for our internal setup / getSolution functions
    const ProcessorGroup*    _pg;
    const PatchSubset*       _patches;
    const MaterialSubset*    _matls;
    DataWarehouse*           _old_dw;
    DataWarehouse*           _new_dw;
    DataWarehouse*           _A_dw;
    DataWarehouse*           _b_dw;
    DataWarehouse*           _guess_dw;

    HYPRE_PtrToSolverFcn*    _precond;
    HYPRE_PtrToSolverFcn*    _pcsetup;

    int _activeInterface;                             // Bit mask: which
                                                      // interfaces are active

    // Hypre ParCSR interface objects
    HYPRE_ParCSRMatrix       _A_Par;                  // Left-hand-side matrix
    HYPRE_ParVector          _b_Par;                  // Right-hand-side vector
    HYPRE_ParVector          _x_Par;                  // Solution vector

  }; // end class HypreDriver


  // Useful functions, printouts

  template<class Types>
    HypreDriver<Types>* newHypreDriver
    (const HypreInterface& interface,
     const Level* level,
     const MaterialSet* matlset,
     const VarLabel* A, Task::WhichDW which_A_dw,
     const VarLabel* x, bool modifies_x,
     const VarLabel* b, Task::WhichDW which_b_dw,
     const VarLabel* guess,
     Task::WhichDW which_guess_dw,
     const HypreSolverParams* params);

  template<class Types>
    typename HypreDriver<Types>::Side&
    operator++(typename HypreDriver<Types>::Side &s);

  template<class Types>
  std::ostream&
    operator << (std::ostream& os,
                 const typename HypreDriver<Types>::CoarseFineViewpoint& v);
  template<class Types>
  std::ostream&
    operator << (std::ostream& os,
                 const typename HypreDriver<Types>::ConstructionStatus& s);
  template<class Types>
  std::ostream&
    operator << (std::ostream& os,
                 const typename HypreDriver<Types>::Side& s);

  // TODO: move this to impICE.cc/BoundaryCond.cc where A is constructed.
  double harmonicAvg(const Point& x,
                     const Point& y,
                     const Point& z,
                     const double& Ax,
                     const double& Ay);
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
