/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverCG
   
   A Hypre CG (conjugate gradient) solver.

GENERAL INFORMATION

   File: HypreSolverCG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HypreSolverCG sets up and destroys the Hypre conjugate gradient
   solver. It can optionally employ a preconditioner.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverCG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverCG(HypreDriver* driver,
                  HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverCG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverCG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h
