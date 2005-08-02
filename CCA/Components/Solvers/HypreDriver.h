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
   HypreDriver, HypreSolverParams, RefCounted, solve, HypreSolverAMR.

DESCRIPTION
   Class HypreDriver is a wrapper for calling Hypre solvers and
   preconditioners. It does not know about the internal Hypre interfaces
   like Struct and SStruct. Instead, it uses the generic HypreInterface
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
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriver_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {

  /* Forward declarations */
  class HypreSolverParams;

  template<class Types>
    class HypreDriver : public RefCounted {

    /*========================== PUBLIC SECTION ==========================*/
  public:

    /*---------- Types ----------*/

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
      {}
    
    virtual ~HypreDriver(void) {}

    void solve(const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               Handle<HypreDriver<Types> >);

    /*========================== PRIVATE SECTION ==========================*/
  private:

    /*---------- Data members ----------*/

    /* Uintah input data */
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

#if 0
    /* Make these functions virtual or incorporate them into a generic
       solver setup function? */
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToSolverFcn& precond,
                      HYPRE_PtrToSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver);
    void destroyPrecond(HYPRE_StructSolver& precond_solver); // TODO: & or not?
#endif

  }; // end class HypreDriver
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
