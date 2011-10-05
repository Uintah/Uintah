/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
#include <Core/Util/RefCounted.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Solvers/HypreTypes.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverBase.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondBase.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ProblemSetupException.h>

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
                const PatchSet* perProcPatches,
                const HypreInterface& interface = HypreInterfaceNA) :
      _level(level), _matlset(matlset),
      _A_label(A), _which_A_dw(which_A_dw),
      _X_label(x), _modifies_x(modifies_x),
      _B_label(b), _which_b_dw(which_b_dw),
      _guess_label(guess), _which_guess_dw(which_guess_dw),
      _params(params), _interface(interface),
      _perProcPatches(perProcPatches)
      {}    
    virtual ~HypreDriver(void) {}
    virtual void cleanup() = 0;

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
    virtual void printMatrix(const string& fileName = "Matrix") = 0;
    virtual void printRHS(const string& fileName = "RHS") = 0;
    virtual void printSolution(const string& fileName = "X") = 0;

    // Generic solve functions
    template<class Types>
      void solve(const ProcessorGroup* pg,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 Handle<HypreDriver>);
                 
   
    virtual void gatherSolutionVector(void) = 0;

    virtual void makeLinearSystem_CC(const int matl);
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
    const PatchSet*          _perProcPatches;
    bool                     _requiresPar; // Solver requires ParCSR format

    // Hypre ParCSR interface objects. Can be used by Struct or
    // SStruct to feed certain solvers. This should maybe be part of a
    // future HypreParCSR interface.
    HYPRE_ParCSRMatrix       _HA_Par;       // Left-hand-side matrix
    HYPRE_ParVector          _HB_Par;       // Right-hand-side vector
    HYPRE_ParVector          _HX_Par;       // Solution vector

  }; // end class HypreDriver

  //========================== Utilities, printouts ==========================
  HypreDriver*    newHypreDriver(const HypreInterface& interface,
                                 const Level* level,
                                 const MaterialSet* matlset,
                                 const VarLabel* A, Task::WhichDW which_A_dw,
                                 const VarLabel* x, bool modifies_x,
                                 const VarLabel* b, Task::WhichDW which_b_dw,
                                 const VarLabel* guess,
                                 Task::WhichDW which_guess_dw,
                                 const HypreSolverParams* params,
                                 const PatchSet* perProcPatches);

  BoxSide         patchFaceSide(const Patch::FaceType& patchFace);

  //_____________________________________________________________________
  // Function HypreDriver::solve~
  // Main solve function.
  //_____________________________________________________________________
  template<class Types>
    void
    HypreDriver::solve(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       Handle<HypreDriver>)

    {
      DebugStream cout_doing("HYPRE_DOING_COUT", false);
      DebugStream cout_dbg("HYPRE_DBG", false);
      int mpiRank = Parallel::getMPIRank();
      cout_doing << mpiRank<< " HypreDriver::solve() BEGIN" << "\n";

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
      cout_dbg << mpiRank << " Checking arguments and parameters ... ";
      SolverType solverType = getSolverType(_params->solverTitle);
      const int numLevels = new_dw->getGrid()->numLevels();
      
      if ((solverType == FAC) && (numLevels < 2)) {
        throw InternalError("FAC solver needs at least 2 levels",
                            __FILE__, __LINE__);
      }

      for(int m = 0; m < matls->size(); m++){
        int matl = matls->get(m);
        double tstart = SCIRun::Time::currentSeconds();
        
       
        // Initialize the preconditioner and solver
        PrecondType precondType   = getPrecondType(_params->precondTitle);
        HyprePrecondBase* precond = newHyprePrecond(precondType);
        
        SolverType solverType = getSolverType(_params->solverTitle);
        HypreSolverBase* solver = newHypreSolver(solverType,this,precond);

        // Set up the preconditioner and tie it to solver
        if (precond) {
          precond->setSolver(solver);
          precond->setup();
        }
        
        //__________________________________
        //Construct Hypre linear system 
        cout_dbg << mpiRank << " Making linear system" << "\n";
        _requiresPar = solver->requiresPar();

        makeLinearSystem_CC(matl);
        
        printMatrix("Matrix");
        printRHS("RHS");
        
        //__________________________________
        // Solve the linear system
      
        cout_dbg << mpiRank << " Solving the linear system" << "\n";
        double solve_start = SCIRun::Time::currentSeconds();
        // Setup & solve phases
        int timeSolve = hypre_InitializeTiming("Solver Setup");
        hypre_BeginTiming(timeSolve);
        
        solver->solve();  
        gatherSolutionVector();
        
        hypre_EndTiming(timeSolve);
        hypre_PrintTiming("Setup phase time", MPI_COMM_WORLD);
        hypre_FinalizeTiming(timeSolve);
        hypre_ClearTiming();
        timeSolve = 0; 
        
        double solve_dt = SCIRun::Time::currentSeconds()-solve_start;
        
        //__________________________________
        // Check if converged,
        const HypreSolverBase::Results& results = solver->getResults();
        double finalResNorm = results.finalResNorm;
        int numIterations = results.numIterations;
        
        if ((finalResNorm > _params->tolerance) ||(finite(finalResNorm) == 0)) {
          if (_params->restart){
            if(pg->myrank() == 0)
              std::cout << "AMRSolver not converged in " << numIterations 
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
        
        //__________________________________
        // Push the solution back into Uintah
        getSolution_CC(matl);
        printSolution("Solution");
                
        double dt = SCIRun::Time::currentSeconds()-tstart;
        if(pg->myrank() == 0){
          std::cerr << "Solve of " << _X_label->getName() 
                    << " on level " << _level->getIndex()
                    << " completed in " << dt 
                    << " seconds (solve only: " << solve_dt 
                    << " seconds, " << numIterations
                    << " iterations, residual=" << finalResNorm << ")\n";
        }
          
        delete solver;
        delete precond;
      } // for m (matls loop)
      cout_doing << mpiRank<<" HypreDriver::solve() END" << "\n";
      cleanup();
    } // end solve()
} // end namespace Uintah

#endif
