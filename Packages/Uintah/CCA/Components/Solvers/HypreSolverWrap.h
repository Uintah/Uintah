/*--------------------------------------------------------------------------
 * File: HypreSolverWrap.h
 *
 * class HypreSolverWrap is a wrapper for calling Hypre solvers and
 * preconditioners.
 *
 * Note: This interface is written for Hypre version 1.9.0b (released 2005)
 * or later.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverWrap_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverWrap_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

// hypre includes
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>

namespace Uintah {

  /* Forward declarations */
  class HypreSolverParams;

  template<class Types>
    class HypreSolverWrap : public RefCounted {
    /*_____________________________________________________________________
      class HypreSolverWrap: solver wrapper for a specific variable type.
      In this class we implement the actual AMR solver call. The class is
      templated against the variable type: cell centered, node centered, etc.
      NOTE: currently only CC is implemented for the pressure solver in 
      implicit AMR ICE.
      _____________________________________________________________________*/
  public:
    HypreSolverWrap(const Level* level,
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
    
    virtual ~HypreSolverWrap() {}

    void solve(const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               Handle<HypreSolverWrap<Types> >);

    /*---------- HypreSolverWrap data members ----------*/
  private:
    const Level* level;
    const MaterialSet* matlset;
    const VarLabel* A_label;
    Task::WhichDW which_A_dw;
    const VarLabel* X_label;
    const VarLabel* B_label;
    Task::WhichDW which_b_dw;
    bool modifies_x;
    const VarLabel* guess_label;
    Task::WhichDW which_guess_dw;
    const HypreSolverParams* params;

    /* Make these functions virtual or incorporate them into a generic
       solver setup function? */
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToSolverFcn& precond,
                      HYPRE_PtrToSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver);
    void destroyPrecond(HYPRE_StructSolver& precond_solver); // TODO: & or not?


  }; // end class HypreSolverWrap
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverWrap_h
