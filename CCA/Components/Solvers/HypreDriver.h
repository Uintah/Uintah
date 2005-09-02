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
  HypreSolverBase, HypreSolverParams, RefCounted, solve, AMRSolver.

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
  HypreSolverBase::newSolver determines the specific Hypre
  interface and solver, based on the parameters in HypreSolverParams.
  HypreDriver::solve() is the task-scheduled function in
  AMRSolver::scheduleSolve() that is activated by
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
#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreTypes.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverBase.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePreconds/HyprePrecondBase.h>

namespace Uintah {

  // Forward declarations
  class HypreSolverParams;
  class VarLabel;
  class Level;
  
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
                const HypreSolverParams* params,
                const HypreInterface& interface = HypreInterfaceNA) :
      _level(level), _matlset(matlset),
      _A_label(A), _which_A_dw(which_A_dw),
      _X_label(x), _modifies_x(modifies_x),
      _B_label(b), _which_b_dw(which_b_dw),
      _guess_label(guess), _which_guess_dw(which_guess_dw),
      _params(params), _interface(interface)
      {}    
    virtual ~HypreDriver(void) {}

    //---------- Data member modifyable access ----------
    // void setInterface(HypreInterface& interface) { _interface = interface; }
    HYPRE_ParCSRMatrix& getAPar(void) { return _HA_Par; }  // LHS
    HYPRE_ParVector&    getBPar(void) { return _HB_Par; }  // RHS
    HYPRE_ParVector&    getXPar(void) { return _HX_Par; }  // Solution

    //---------- Data member unmodifyable access ----------
    const HypreSolverParams*  getParams(void) const { return _params; }
    const ProcessorGroup*     getPG(void) const { return _pg; }
    const PatchSubset*        getPatches(void) { return _patches; }
    const HypreInterface&     getInterface(void) const { return _interface; }
    const HYPRE_ParCSRMatrix& getAPar(void) const { return _HA_Par; }
    const HYPRE_ParVector&    getBPar(void) const { return _HB_Par; }
    const HYPRE_ParVector&    getXPar(void) const { return _HX_Par; }
    
    // Utilities, HYPRE data printouts
    bool         isConvertable(const HypreInterface& to);
    virtual void printMatrix(const string& fileName = "output") = 0;
    virtual void printRHS(const string& fileName = "output_b") = 0;
    virtual void printSolution(const string& fileName = "output_x") = 0;

    // Generic solve functions (common for all var types)
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
    virtual void gatherSolutionVector(void) = 0;

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

    HypreInterface           _interface;   // Hypre interface currently in use
    bool                     _requiresPar; // Solver requires ParCSR format

    // Hypre ParCSR interface objects. Can be used by Struct or
    // SStruct to feed certain solvers. This should maybe be part of a
    // future HypreParCSR interface.
    HYPRE_ParCSRMatrix       _HA_Par;       // Left-hand-side matrix
    HYPRE_ParVector          _HB_Par;       // Right-hand-side vector
    HYPRE_ParVector          _HX_Par;       // Solution vector

  }; // end class HypreDriver

  //========================== Utilities, printouts ==========================

  template<class Types>
    TypeDescription::Type TypeTemplate2Enum(const Types& t);
  // Specific instantiations
  template<>
    TypeDescription::Type TypeTemplate2Enum(const SFCXTypes& t);
  template<>
    TypeDescription::Type TypeTemplate2Enum(const SFCYTypes& t);
  template<>
    TypeDescription::Type TypeTemplate2Enum(const SFCZTypes& t);
  template<>
    TypeDescription::Type TypeTemplate2Enum(const NCTypes& t);
  template<>
    TypeDescription::Type TypeTemplate2Enum(const CCTypes& t);

  HypreDriver*    newHypreDriver(const HypreInterface& interface,
                                 const Level* level,
                                 const MaterialSet* matlset,
                                 const VarLabel* A, Task::WhichDW which_A_dw,
                                 const VarLabel* x, bool modifies_x,
                                 const VarLabel* b, Task::WhichDW which_b_dw,
                                 const VarLabel* guess,
                                 Task::WhichDW which_guess_dw,
                                 const HypreSolverParams* params);

  HypreInterface& operator ++ (HypreInterface& interface);
  ostream&        operator << (ostream& os, const HypreInterface& i);
  
  BoxSide&        operator ++ (BoxSide& side);
  ostream&        operator << (ostream& os, const BoxSide& s);
  BoxSide         patchFaceSide(const Patch::FaceType& patchFace);

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
      cerr << "HypreDriver::solve() BEGIN" << "\n";
      double tstart = SCIRun::Time::currentSeconds();

      // Assign HypreDriver references that are convenient to have in
      // makeLinearSystem(), getSolution().
      _pg = pg;
      _patches = patches;
      _matls = matls;
      _old_dw = old_dw;
      _new_dw = new_dw;
      _A_dw = new_dw->getOtherDataWarehouse(_which_A_dw);
      _b_dw = new_dw->getOtherDataWarehouse(_which_b_dw);
      _guess_dw = new_dw->getOtherDataWarehouse(_which_guess_dw);
    
      // Check parameter correctness
      cerr << "Checking arguments and parameters ... ";
      SolverType solverType = getSolverType(_params->solverTitle);
      const int numLevels = new_dw->getGrid()->numLevels();
      cerr << "numLevels = " << numLevels << "\n";
      cerr << "solverType = " << solverType << "\n";
      if ((solverType == FAC) && (numLevels < 2)) {
        throw InternalError("FAC solver needs at least 2 levels",
                            __FILE__, __LINE__);
      }

      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        cerr << "Doing m = " << m << "/" << matls->size()
             << "  matl = " << matl << "\n";

        // Initialize the preconditioner
        cerr << "Creating preconditioner" << "\n";
        PrecondType precondType = getPrecondType(_params->precondTitle);
        cerr << "precondType = " << precondType << "\n";
        HyprePrecondBase* precond = newHyprePrecond(precondType);
        // Construct Hypre solver object that uses the hypreInterface we
        // chose. Specific solver object is arbitrated in
        // HypreSolverBase. The solver is linked to the HypreDriver
        // data and the preconditioner.
        cerr << "Creating solver" << "\n";
        SolverType solverType = getSolverType(_params->solverTitle);
        cerr << "solverType = " << solverType << "\n";
        HypreSolverBase* solver = newHypreSolver(solverType,this,precond);

        // Set up the preconditioner and tie it to solver
        if (precond) {
          cerr << "Setting up preconditioner" << "\n";
          precond->setSolver(solver);
          precond->setup();
        }
#if 0

        // Construct Hypre linear system for the specific variable type
        // and Hypre interface
        _requiresPar = solver->requiresPar();
        cerr << "Making linear system" << "\n";
        makeLinearSystem<Types>(matl);
    
        //-----------------------------------------------------------
        // Solve the linear system
        //-----------------------------------------------------------
        cerr << "Solving the linear system" << "\n";
        double solve_start = SCIRun::Time::currentSeconds();
        // Setup & solve phases
        int timeSolve = hypre_InitializeTiming("Solver Setup");
        hypre_BeginTiming(timeSolve);
        solver->solve();  // Currently depends on A, b, x
        gatherSolutionVector();
        hypre_EndTiming(timeSolve);
        hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
        hypre_FinalizeTiming(timeSolve);
        hypre_ClearTiming();
        timeSolve = 0; // to eliminate unused warning

        double solve_dt = SCIRun::Time::currentSeconds()-solve_start;

        //-----------------------------------------------------------
        // Check if converged, print solve statistics
        //-----------------------------------------------------------
        cerr << "Print solve statistics" << "\n";
        const HypreSolverBase::Results& results = solver->getResults();
        double finalResNorm = results.finalResNorm;
        int numIterations = results.numIterations;
        if ((finalResNorm > _params->tolerance) ||
            (finite(finalResNorm) == 0)) {
          if (_params->restart){
            if(pg->myrank() == 0)
              cerr << "AMRSolver not converged in " << numIterations 
                   << " iterations, final residual= " << finalResNorm
                   << ", requesting smaller timestep\n";
            //new_dw->abortTimestep();
            //new_dw->restartTimestep();
          } else {
            throw ConvergenceFailure("AMRSolver variable: "
                                     +_X_label->getName()+", solver: "
                                     +_params->solverTitle+", preconditioner: "
                                     +_params->precondTitle,
                                     numIterations, finalResNorm,
                                     _params->tolerance,__FILE__,__LINE__);
          }
        } // if (finalResNorm is ok)
#endif
        /* Get the solution x values back into Uintah */
        cerr << "Calling getSolution" << "\n";
        getSolution<Types>(matl);

#if 0
        /*-----------------------------------------------------------
         * Print the solution and other info
         *-----------------------------------------------------------*/
        cerr << "Print the solution vector" << "\n";
        printMatrix("output_A");
        printRHS("output_b");
        printSolution("output_x1");
        cerr << "Iterations = " << numIterations << "\n";
        cerr << "Final Relative Residual Norm = "
             << finalResNorm << "\n";
        cerr << "" << "\n";
      
        double dt = SCIRun::Time::currentSeconds()-tstart;
        if(pg->myrank() == 0){
          cerr << "Solve of " << _X_label->getName() 
               << " on level " << _level->getIndex()
               << " completed in " << dt 
               << " seconds (solve only: " << solve_dt 
               << " seconds, " << numIterations
               << " iterations, residual=" << finalResNorm << ")\n";
        }
        tstart = SCIRun::Time::currentSeconds();
#endif
        delete solver;
        delete precond;
      } // for m (matls loop)
      cerr << "HypreDriver::solve() END" << "\n";
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

    } // end getSolution()

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriver_h
