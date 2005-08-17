#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriver_h

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

#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>

namespace Uintah {

  // Forward declarations
  class HypreSolverParams;
  
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
      params(params), _interface(0)
      {}    
    virtual ~HypreDriver(void);

    // Data member modifyable access
    HYPRE_StructMatrix& getA(void) { return _HA; }  // LHS
    HYPRE_StructVector& getB(void) { return _HB; }  // RHS
    HYPRE_StructVector& getX(void) { return _HX; }  // Solution

    // Generic solve functions
    template<class Types>
      void solve(const ProcessorGroup* pg,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 Handle<HypreDriver>);
    template<class Types>
      void makeLinearSystem(const int matl);
    template<class Types>
      void getSolution(const int matl);
    
    // Set up linear system, read back solution for each variable type
    // (default: throw exception; if implemented in derived classes
    // from HypreDriver, it will be activated from the code therein)
    virtual void makeLinearSystem_SFCX(const int matl);
    virtual void makeLinearSystem_SFCY(const int matl);
    virtual void makeLinearSystem_SFCZ(const int matl);
    virtual void makeLinearSystem_NC(const int matl);
    virtual void makeLinearSystem_CC(const int matl);

    virtual void getSolution_SFCX(const int matl);
    virtual void getSolution_SFCY(const int matl);
    virtual void getSolution_SFCZ(const int matl);
    virtual void getSolution_NC(const int matl);
    virtual void getSolution_CC(const int matl);

    virtual void gatherSolutionVector(void) = 0;

    // HYPRE data printouts
    virtual void printMatrix(const string& fileName = "output") = 0;
    virtual void printRHS(const string& fileName = "output_b") = 0;
    virtual void printSolution(const string& fileName = "output_x") = 0;

    //========================== PROTECTED SECTION ==========================
  protected:

    //---------- Data members ----------
    // Uintah input data
    const Level*             _level;
    const MaterialSet*       _matlset;
    const VarLabel*          _A_label;
    Task::WhichDW            _which_A_dw;
    const VarLabel*          _X_label;
    bool                     _modifies_x;
    const VarLabel*          _B_label;
    Task::WhichDW            _which_b_dw;
    const VarLabel*          _guess_label;
    Task::WhichDW            _which_guess_dw;
    const HypreSolverParams* _params;

    // Uintah variables assigned inside solve() for our internal setup
    // / getSolution functions
    const ProcessorGroup*    _pg;
    const PatchSubset*       _patches;
    const MaterialSubset*    _matls;
    DataWarehouse*           _old_dw;
    DataWarehouse*           _new_dw;
    DataWarehouse*           _A_dw;
    DataWarehouse*           _b_dw;
    DataWarehouse*           _guess_dw;

    HypreInterface           _interface;  // Hypre interface currently in use

    // TODO: create a HypreParCSR interface that can be constructed
    // from a struct or sstruct hypre driver object.
    // Hypre ParCSR interface objects
    //    HYPRE_ParCSRMatrix       _A_Par;                  // Left-hand-side matrix
    //    HYPRE_ParVector          _b_Par;                  // Right-hand-side vector
    //    HYPRE_ParVector          _x_Par;                  // Solution vector

  }; // end class HypreDriver


  //========================== Utilities, printouts ==========================

  template<class Types>
    TypeDescription::Type TypeTemplate2Enum(const Types& t);

  HypreDriver* newHypreDriver
    (const HypreInterface& interface,
     const Level* level,
     const MaterialSet* matlset,
     const VarLabel* A, Task::WhichDW which_A_dw,
     const VarLabel* x, bool modifies_x,
     const VarLabel* b, Task::WhichDW which_b_dw,
     const VarLabel* guess,
     Task::WhichDW which_guess_dw,
     const HypreSolverParams* params);
  HypreInterface& operator ++ (HypreInterface& i);
  BoxSide&        operator ++ (BoxSide& i);
  std::ostream&   operator << (std::ostream& os,
                               const BoxSide& s);

    // Utilities
    void printValues(const int stencilSize,
                     const int numCells,
                     const double* values = 0,
                     const double* rhsValues = 0,
                     const double* solutionValues = 0);

  // TODO: move this to impICE.cc/BoundaryCond.cc where A is constructed.
  double harmonicAvg(const Point& x,
                     const Point& y,
                     const Point& z,
                     const double& Ax,
                     const double& Ay);

  //========================================================================
  // Implementation of the templated part of class HypreDriver
  //========================================================================

  template<class Types>
  void
  HypreDriver::solve(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     Handle<HypreDriver>)
    //_____________________________________________________________________
    // Function HypreDriver::solve~
    // Main solve function.
    //_____________________________________________________________________
  {
    using namespace SCIRun;
    double tstart = Time::currentSeconds();

    // Assign HypreDriver references that are convenient to have in
    // makeLinearSystem(), getSolution().
    _pg = pg;
    _patches = patches;
    _matls = matls;
    _old_dw = old_dw;
    _new_dw = new_dw;
    _A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
    _b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
    _guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
    // Check parameter correctness
    cerr << "Checking arguments and parameters ... ";
    SolverType solverType = getSolverType(params->solverTitle);
    const int numLevels = new_dw->getGrid()->numLevels();
    if ((solverType == FAC) && (numLevels < 2)) {
      throw InternalError("FAC solver needs at least 2 levels",
                          __FILE__, __LINE__);
    }

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      /* Construct Hypre linear system for the specific variable type
         and Hypre interface */
      makeLinearSystem<Types>(matl);
    
      /* Construct Hypre solver object that uses the hypreInterface we
         chose. Specific solver object is arbitrated in HypreGenericSolver
         according to param->solverType. */
      SolverType solverType = getSolverType(params->solverTitle);
      HypreGenericSolver* solver = 
        newHypreGenericSolver(solverType,this);

      //-----------------------------------------------------------
      // Solve the linear system
      //-----------------------------------------------------------
      double solve_start = Time::currentSeconds();

      // Setup phase
      int timeSetup = hypre_InitializeTiming("Solver Setup");
      hypre_BeginTiming(timeSetup);
      solver->setup(this);  // Depends only on A
      hypre_EndTiming(timeSetup);
      hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(timeSetup);
      hypre_ClearTiming();
      timeSetup = 0; // to eliminate unused warning

      // Solve phase
      int timeSolve = hypre_InitializeTiming("Solver Setup");
      hypre_BeginTiming(timeSolve);
      solver->solve(this);  // Depends on A and b
      gatherSolutionVector();
      hypre_EndTiming(timeSolve);
      hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(timeSolve);
      hypre_ClearTiming();
      timeSolve = 0; // to eliminate unused warning

      double solve_dt = Time::currentSeconds()-solve_start;

      //-----------------------------------------------------------
      // Check if converged, print solve statisticsS
      //-----------------------------------------------------------
      const HypreGenericSolver::Results& results = solver->getResults();
      double finalResNorm = results.finalResNorm;
      int numIterations = results.numIterations;
      if ((finalResNorm > params->tolerance) ||
          (finite(finalResNorm) == 0)) {
        if (params->restart){
          if(pg->myrank() == 0)
            cerr << "HypreSolver not converged in " << numIterations 
                 << "iterations, final residual= " << finalResNorm
                 << ", requesting smaller timestep\n";
          //new_dw->abortTimestep();
          //new_dw->restartTimestep();
        } else {
	  throw ConvergenceFailure("HypreSolver variable: "
                                   +X_label->getName()+", solver: "
                                   +params->solverTitle+", preconditioner: "
                                   +params->precondTitle,
				   numIterations, finalResNorm,
				   params->tolerance,__FILE__,__LINE__);
        }
      } // if (finalResNorm is ok)

      /* Get the solution x values back into Uintah */
      getSolution<Types>(matl);

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/
      std::cerr << "Print the solution vector" << "\n";
      printSolution("output_x1");
      std::cerr << "Iterations = " << numIterations << "\n";
      std::cerr << "Final Relative Residual Norm = "
           << finalResNorm << "\n";
      std::cerr << "" << "\n";
      
      delete solver;

      double dt=Time::currentSeconds()-tstart;
      if(pg->myrank() == 0){
        cerr << "Solve of " << X_label->getName() 
             << " on level " << level->getIndex()
             << " completed in " << dt 
             << " seconds (solve only: " << solve_dt 
             << " seconds, " << numIterations
             << " iterations, residual=" << finalResNorm << ")\n";
      }
      tstart = Time::currentSeconds();
    } // for m (matls loop)
  } // end solve() for

  template<class Types>
  void
  HypreDriver::makeLinearSystem(const int matl)
    //_____________________________________________________________________
    // Function HypreDriver::makeLinearSystem~
    // Switch between makeLinearSystems of specific variable types by
    // template class.
    //_____________________________________________________________________
  {
    Types t;
    TypeDescription::Type domType = TypeTemplate2Enum(t);
    switch (domType) {
    case TypeDescription::SFCXVariable:
      {
        makeLinearSystem_SFCX(matl);
        break;
      } // end case SFCXVariable 

    case TypeDescription::SFCYVariable:
      {
        makeLinearSystem_SFCY(matl);
        break;
      } // end case SFCYVariable 

    case TypeDescription::SFCZVariable:
      {
        makeLinearSystem_SFCZ(matl);
        break;
      } // end case SFCZVariable 

    case TypeDescription::NCVariable:
      {
        makeLinearSystem_NC(matl);
        break;
      } // end case NCVariable 

    case TypeDescription::CCVariable:
      {
        makeLinearSystem_CC(matl);
        break;
      } // end case CCVariable

    default:
      {
        throw InternalError("Unknown variable type in scheduleSolve",
                            __FILE__, __LINE__);
      } // end default

    } // end switch (domType)

  } // end makeLinearSystem() for


  template<class Types>
  void
  HypreDriver::getSolution(const int matl)
    //_____________________________________________________________________
    // Function HypreDriver::getSolution~
    // Switch between getSolutions of specific variable types by
    // template class.
    //_____________________________________________________________________
  {
    Types t;
    TypeDescription::Type domType = TypeTemplate2Enum(t);
    switch (domType) {
    case TypeDescription::SFCXVariable:
      {
        getSolution_SFCX(matl);
        break;
      } // end case SFCXVariable 

    case TypeDescription::SFCYVariable:
      {
        getSolution_SFCY(matl);
        break;
      } // end case SFCYVariable 

    case TypeDescription::SFCZVariable:
      {
        getSolution_SFCZ(matl);
        break;
      } // end case SFCZVariable 

    case TypeDescription::NCVariable:
      {
        getSolution_NC(matl);
        break;
      } // end case NCVariable 

    case TypeDescription::CCVariable:
      {
        getSolution_CC(matl);
        break;
      } // end case CCVariable

    default:
      {
        throw InternalError("Unknown variable type in scheduleSolve",
                            __FILE__, __LINE__);
      } // end default

    } // end switch (domType)

  } // end getSolution() for


} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
