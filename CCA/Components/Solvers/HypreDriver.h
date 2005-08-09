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

  using std::cerr;

  // Forward declarations
  class HypreSolverParams;

  template<class Types>
    class HypreDriver : public RefCounted {

    //========================== PUBLIC SECTION ==========================
  public:

    //---------- Types ----------

    enum InterfaceType {       // Hypre system interface for the solver
      Struct, SStruct, ParCSR
    };
    
    //---------- Construction & Destruction ----------

    HypreDriver(const Level* level,
                const MaterialSet* matlset,
                const VarLabel* A, Task::WhichDW which_A_dw,
                const VarLabel* x, bool modifies_x,
                const VarLabel* b, Task::WhichDW which_b_dw,
                const VarLabel* guess,
                Task::WhichDW which_guess_dw,
                const HypreSolverParams* params)
      : level(level), matlset(matlset),
      A_label(A), which_A_dw(which_A_dw),
      X_label(x), 
      B_label(b), which_b_dw(which_b_dw),
      modifies_x(modifies_x),
      guess_label(guess), which_guess_dw(which_guess_dw), params(params)
      {
        // No interfaces are currently active
        _active.clear();
        _active[Struct ] = false;
        _active[SStruct] = false;
        _active[ParCSR ] = false;
      }
    
    virtual ~HypreDriver(void)
      {
        cerr << "Destroying Solver object" << "\n";
        /* Destroy graph objects */
        /* Destroy matrix, RHS, solution objects */

        if (_active[Struct]) {
          cerr << "Destroying Struct matrix, RHS, solution objects" << "\n";
          HYPRE_StructMatrixDestroy(_A_Struct);
          HYPRE_StructVectorDestroy(_b_Struct);
          HYPRE_StructVectorDestroy(_x_Struct);
        }
        if (_active[SStruct]) {
          cerr << "Destroying SStruct matrix, RHS, solution objects" << "\n";
          HYPRE_SStructMatrixDestroy(_A_SStruct);
          HYPRE_SStructVectorDestroy(_b_SStruct);
          HYPRE_SStructVectorDestroy(_x_SStruct);
          cerr << "Destroying graph objects" << "\n";
          HYPRE_SStructGraphDestroy(_graph_SStruct);
        }
        if (_active[ParCSR]) {
          cerr << "Destroying ParCSR matrix, RHS, solution objects" << "\n";
          HYPRE_ParCSRMatrixDestroy(_A_Par);
          HYPRE_ParVectorDestroy(_b_Par);
          HYPRE_ParVectorDestroy(_x_Par);
        }
      }

    void solve(const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               Handle<HypreDriver<Types> >);

    void convertData(const InterfaceType& fromInterface,
                     const InterfaceType& toInterface);

    //========================== PRIVATE SECTION ==========================
  private:

    //---------- Types ----------

    enum CoarseFineViewpoint {
      CoarseToFine,
      FineToCoarse
    };
    
    enum ConstructionStatus {
      Graph,
      Matrix
    };

    //================================================================
    // Common to all interfaces and variable types
    //================================================================

    void initialize(const InterfaceType& fromInterface);
    void initializeData(const InterfaceType& fromInterface);
    void assemble(const InterfaceType& fromInterface);

    // Utilities
    void printValues(const Patch* patch,
                     const int stencilSize,
                     const int numCells,
                     const double* values = 0,
                     const double* rhsValues = 0,
                     const double* solutionValues = 0);

    //================================================================
    // Implementation of linear system construction for specific Hypre
    // interfaces and variable types
    //================================================================
    template<class Types>
      void makeLinearSystemStruct(void);
    template<class Types>
      void makeLinearSystemSStruct(void);
    template<class Types>
      void getSolutionStruct(void);
    template<class Types>
      void getSolutionSStruct(void);

    // SStruct C/F graph & matrix construction
    void makeConnections(const ConstructionStatus& status,
                         const Hierarchy& hier,
                         const HYPRE_SStructStencil& stencil,
                         const Counter level,
                         const Patch* patch,
                         const Counter& d,
                         const Side& s,
                         const CoarseFineViewpoint& viewpoint);

    // SStruct Graph construction
    void makeGraph(const Hierarchy& hier,
                   const HYPRE_SStructGrid& grid,
                   const HYPRE_SStructStencil& stencil);
    
    // SStruct matrix construction
    void makeInteriorEquations(const Counter level,
                               const Hierarchy& hier,
                               const HYPRE_SStructGrid& grid,
                               const HYPRE_SStructStencil& stencil);
    void makeUnderlyingIdentity(const Counter level,
                                const HYPRE_SStructStencil& stencil,
                                const Box& coarseUnderFine);

#if 0
    // Make these functions virtual or incorporate them into a generic
    //   solver setup function?
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToSolverFcn& precond,
                      HYPRE_PtrToSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver);
    void destroyPrecond(HYPRE_StructSolver& precond_solver); // TODO:
                                                             // use &
                                                             // or
                                                             // not?
#endif

    //---------- Data members ----------
    // Uintah input data
    const Level*             level;
    const MaterialSet*       matlset;
    const VarLabel*          A_label;
    Task::WhichDW            which_A_dw;
    const VarLabel*          X_label;
    const VarLabel*          B_label;
    Task::WhichDW            which_b_dw;
    bool                     modifies_x;
    const VarLabel*          guess_label;
    Task::WhichDW            which_guess_dw;
    const HypreSolverParams* params;

    std::map<InterfaceType,bool> _active;             // Which interfaces
                                                      // are active

    // Hypre Struct interface objects
    HYPRE_StructGrid        _grid_Struct;             // level&patch hierarchy
    HYPRE_StructMatrix       _A_Struct;               // Left-hand-side matrix
    HYPRE_StructVector       _b_Struct;               // Right-hand-side vector
    HYPRE_StructVector       _x_Struct;               // Solution vector

    // Hypre SStruct interface objects
    HYPRE_SStructGrid        _grid_SStruct;           // level&patch hierarchy
    HYPRE_SStructStencil     _stencil_SStruct;        // Same stencil @ all levels
    HYPRE_SStructMatrix      _A_SStruct;              // Left-hand-side matrix
    HYPRE_SStructVector      _b_SStruct;              // Right-hand-side vector
    HYPRE_SStructVector      _x_SStruct;              // Solution vector
    HYPRE_SStructGraph       _graph_SStruct;          // Unstructured
                                                      // connection graph
    // Hypre ParCSR interface objects
    HYPRE_ParCSRMatrix       _A_Par;                  // Left-hand-side matrix
    HYPRE_ParVector          _b_Par;                  // Right-hand-side vector
    HYPRE_ParVector          _x_Par;                  // Solution vector

  }; // end class HypreDriver

  std::ostream&
    operator << (std::ostream& os, const HypreDriver::CoarseFineViewpoint& v);
  std::ostream&
    operator << (std::ostream& os, const HypreDriver::ConstructionStatus& s);
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
