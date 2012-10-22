/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreDriverStruct_h
#define Packages_Uintah_CCA_Components_Solvers_HypreDriverStruct_h

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
  

KEYWORDS
   HYPRE_Struct, HYPRE_SStruct, HYPRE_ParCSR,
   HypreSolverBase, HypreSolverParams.

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

#include <CCA/Components/Solvers/AMR/HypreDriver.h>

namespace Uintah {

  // Forward declarations
  class HypreSolverParams;

  class HypreDriverStruct : public HypreDriver {
    
    
  public:

    // Construction & Destruction
    HypreDriverStruct(const Level* level,
                      const MaterialSet* matlset,
                      const VarLabel* A, Task::WhichDW which_A_dw,
                      const VarLabel* x, bool modifies_x,
                      const VarLabel* b, Task::WhichDW which_b_dw,
                      const VarLabel* guess,
                      Task::WhichDW which_guess_dw,
                      const HypreSolverParams* params,
                      const PatchSet* perProcPatches,
                      const HypreInterface& interface = HypreInterfaceNA) :
      HypreDriver(level,matlset,A,which_A_dw,x,modifies_x,
                  b,which_b_dw,guess,which_guess_dw,params, perProcPatches, interface) {}
    virtual ~HypreDriverStruct(void);
    virtual void cleanup();

    // Data member modifyable access
    HYPRE_StructMatrix& getA(void) { return _HA; }  // LHS
    HYPRE_StructVector& getB(void) { return _HB; }  // RHS
    HYPRE_StructVector& getX(void) { return _HX; }  // Solution

    // Data member unmodifyable access
    const HYPRE_StructMatrix& getA(void) const { return _HA; }  // LHS
    const HYPRE_StructVector& getB(void) const { return _HB; }  // RHS
    const HYPRE_StructVector& getX(void) const { return _HX; }  // Solution

    // Common for all var types
    virtual void gatherSolutionVector(void);

    // CC variables: 
    virtual void makeLinearSystem_CC(const int matl);
    virtual void getSolution_CC(const int matl);

    // HYPRE data printouts
    virtual void printMatrix(const string& fileName = "output");
    virtual void printRHS(const string& fileName = "output_b");
    virtual void printSolution(const string& fileName = "output_x");

   
  private:

    //---------- Data members ----------
    // Hypre Struct interface objects
    HYPRE_StructGrid         _grid;              // level&patch hierarchy
    HYPRE_StructStencil      _stencil;           // stencil pattern
    HYPRE_StructMatrix       _HA;                // Left-hand-side matrix
    HYPRE_StructVector       _HB;                // Right-hand-side vector
    HYPRE_StructVector       _HX;                // Solution vector

  }; 
} 

#endif 
