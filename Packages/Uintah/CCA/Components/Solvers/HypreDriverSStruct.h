#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriverSStruct_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriverSStruct_h

/*--------------------------------------------------------------------------
  CLASS
  HypreDriverSStruct
   
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
  HypreGenericSolver, HypreSolverParams.

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

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreTypes.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <iostream>

namespace Uintah {
  // Forward declarations
  class Level;
  class Patch;
  class VarLabel;
  class HypreSolverParams;

  class HypreDriverSStruct : public HypreDriver {

    //========================== PUBLIC SECTION ==========================
  public:

    //---------- Construction & Destruction ----------
    HypreDriverSStruct(const Level* level,
                       const MaterialSet* matlset,
                       const VarLabel* A, Task::WhichDW which_A_dw,
                       const VarLabel* x, bool modifies_x,
                       const VarLabel* b, Task::WhichDW which_b_dw,
                       const VarLabel* guess,
                       Task::WhichDW which_guess_dw,
                       const HypreSolverParams* params,
                       const HypreInterface& interface = HypreInterfaceNA) :
      HypreDriver(level,matlset,A,which_A_dw,x,modifies_x,
                  b,which_b_dw,guess,which_guess_dw,params,interface) {}
    virtual ~HypreDriverSStruct(void);

 
    //---------- Data member modifyable access ----------
    HYPRE_SStructMatrix& getA(void) { return _HA; }  // LHS
    HYPRE_SStructVector& getB(void) { return _HB; }  // RHS
    HYPRE_SStructVector& getX(void) { return _HX; }  // Solution

    //---------- Data member unmodifyable access ----------
    const HYPRE_SStructMatrix& getA(void) const { return _HA; }  // LHS
    const HYPRE_SStructVector& getB(void) const { return _HB; }  // RHS
    const HYPRE_SStructVector& getX(void) const { return _HX; }  // Solution

    //---------- Common for all var types ----------
    virtual void printMatrix(const string& fileName = "output");
    virtual void printRHS(const string& fileName = "output_b");
    virtual void printSolution(const string& fileName = "output_x");
    virtual void gatherSolutionVector(void);

    //---------- CC Variables implementation ----------
    virtual void makeLinearSystem_CC(const int matl);
    virtual void getSolution_CC(const int matl);

    //========================== PRIVATE SECTION ==========================
  private:

    //---------- Common for all var types ----------
    enum CoarseFineViewpoint
      // When constructing graph/matrix entries, is the current patch
      // the coarse or the fine one at a C/F boundary?
      {
        DoingCoarseToFine,
        DoingFineToCoarse
      };

    class HyprePatch {
      //___________________________________________________________________
      // class HyprePatch~
      // A convenient structure that holds a Uintah patch grid &
      // geometry data that the Hypre SStruct interface uses.
      //___________________________________________________________________
    public:
      HyprePatch(const Patch* patch,
                 const int matl);  // Construction from Uintah patch
      virtual ~HyprePatch(void);
      const Patch* getPatch(void) const { return _patch; }
      int          getLevel(void) const { return _level; }

      //########## Generic variable type implementation ##########
      // Grid construction
      virtual void addToGrid(HYPRE_SStructGrid& grid,
                             HYPRE_SStructVariable* vars) = 0;

      // C/F graph connections construction
      virtual void makeConnections(HYPRE_SStructGraph& graph,
                                   const CoarseFineViewpoint& viewpoint) = 0;

      // Interior matrix connections construction
      virtual void makeInteriorEquations(HYPRE_SStructMatrix& HA,
                                         DataWarehouse* _A_dw,
                                         const VarLabel* _A_label,
                                         const int stencilSize,
                                         const bool symmetric) = 0;

      // Interior matrix connections construction
      virtual void makeInteriorVector(HYPRE_SStructVector& HV,
                                      DataWarehouse* _V_dw,
                                      const VarLabel* _V_label) = 0;

      // C/F matrix connections construction
      virtual void makeConnections(HYPRE_SStructMatrix& HA,
                                   DataWarehouse* _A_dw,
                                   const VarLabel* _A_label,
                                   const int stencilSize,
                                   const CoarseFineViewpoint& viewpoint) = 0;

    protected:
      //########## Data members ##########
      const Patch* _patch;   // Uintah patch pointer
      int          _matl;    // Material #
      int          _level;   // Patch belong to this level
      IntVector    _low;     // Lower-left interior cell
      IntVector    _high;    // Upper-right interior cell
    }; // end class HyprePatch
    
    //---------- CC Variables implementation ----------
    class HyprePatch_CC : public HyprePatch {
      //___________________________________________________________________
      // class HyprePatch_CC~
      // A convenient structure that holds a Uintah patch grid &
      // geometry data that the Hypre SStruct interface uses, for CC
      // variable solvers.
      //___________________________________________________________________
    public:
      HyprePatch_CC(const Patch* patch,
                    const int matl) :
        HyprePatch(patch,matl) {}
      virtual ~HyprePatch_CC(void) {}

      // Grid construction
      virtual void addToGrid(HYPRE_SStructGrid& grid,
                             HYPRE_SStructVariable* vars);

      // C/F graph connections construction
      virtual void makeConnections(HYPRE_SStructGraph& graph,
                                   const CoarseFineViewpoint& viewpoint);

      // Interior matrix connections construction
      virtual void makeInteriorEquations(HYPRE_SStructMatrix& HA,
                                         DataWarehouse* _A_dw,
                                         const VarLabel* _A_label,
                                         const int stencilSize,
                                         const bool symmetric = false);

      // Interior matrix connections construction
      virtual void makeInteriorVector(HYPRE_SStructVector& HV,
                                      DataWarehouse* _V_dw,
                                      const VarLabel* _V_label);

      // C/F matrix connections construction
      virtual void makeConnections(HYPRE_SStructMatrix& HA,
                                   DataWarehouse* _A_dw,
                                   const VarLabel* _A_label,
                                   const int stencilSize,
                                   const CoarseFineViewpoint& viewpoint);

    }; // end class HyprePatch_CC

    friend std::ostream& operator<< (std::ostream& os,
                                      const CoarseFineViewpoint& v);
    friend std::ostream& operator<< (std::ostream& os,
                                     const HyprePatch& p);

    //---------- Data members ----------
    // Hypre SStruct interface objects
    HYPRE_SStructGrid        _grid;         // level&patch hierarchy
    HYPRE_SStructVariable*   _vars;         // Types of Hypre variables used
    HYPRE_SStructStencil     _stencil;      // Same stencil@all levls
    HYPRE_SStructMatrix      _HA;           // Left-hand-side matrix
    HYPRE_SStructVector      _HB;           // Right-hand-side vector
    HYPRE_SStructVector      _HX;           // Solution vector
    HYPRE_SStructGraph       _graph;        // Unstructured connection graph
    int                      _stencilSize;  // # entries in stencil
  }; // end class HypreDriverSStruct

} // end namespace Uintah

//========================== Utilities, printouts ==========================
void          printLine(const std::string& s, const unsigned int len);

#endif // Packages_Uintah_CCA_Components_Solvers_HypreDriverSStruct_h
