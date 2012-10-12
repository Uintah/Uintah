/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverSMG
   
   A Hypre SMG (geometric multigrid #1) solver.

GENERAL INFORMATION

   File: HypreSolverSMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondSMG sets up and destroys the Hypre SMG
   solver to be used with Hypre solvers. SMG is a geometric multigrid
   solver well suited for Poisson or a diffusion operator with a
   smooth diffusion coefficient.  SMG is used as to solve or as a
   preconditioner is often used with CG or GMRES.

WARNING
      Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverSMG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverSMG(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverSMG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverSMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h
