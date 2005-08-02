/*--------------------------------------------------------------------------
 * File: HypreDriver.h
 *
 * class HypreDriver is a wrapper for calling Hypre solvers and
 * preconditioners. It contains all Hypre interface data types:
 * Struct, SStruct, ParCSR. Only one of them is activated per solver.
 *
 * Note: This interface is written for Hypre version 1.9.0b (released 2005)
 * or later.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriver_h

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
    class HypreDriver : public RefCounted {
    /*_____________________________________________________________________
      class HypreDriver: solver wrapper for a specific variable type.
      In this class we implement the actual AMR solver call. The class is
      templated against the variable type: cell centered, node centered, etc.
      NOTE: currently only CC is implemented for the pressure solver in 
      implicit AMR ICE.
      _____________________________________________________________________*/
  public:

    /*---------- Types ----------*/

    enum HypreInterface {       // Hypre system interface for the solver
      Struct, SStruct, ParCSR
    };

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

    void convertData(const HypreInterface& fromInterface,
                     const HypreInterface& toInterface);


  private:
    /*---------- Data members ----------*/

    /* Uintah input data */
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

    HypreInterface        _hypreInterface; // Currently active type

    /* Struct objects */
    HYPRE_StructMatrix    _StructA;
    HYPRE_StructVector    _StructB;
    HYPRE_StructVector    _StructX;

    /* SStruct objects */
    HYPRE_SStructMatrix   _SStructA;
    HYPRE_SStructVector   _SStructB;
    HYPRE_SStructVector   _SStructX;
    HYPRE_SStructGraph    _SStructGraph;
    
    /* ParCSR objects */
    HYPRE_ParCSRMatrix    _ParA;
    HYPRE_ParVector       _ParB;
    HYPRE_ParVector       _ParX;

    void makeLinearSystem(void); // Depends on Types

    /* Make these functions virtual or incorporate them into a generic
       solver setup function? */
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToSolverFcn& precond,
                      HYPRE_PtrToSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver);
    void destroyPrecond(HYPRE_StructSolver& precond_solver); // TODO: & or not?


  }; // end class HypreDriver
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
