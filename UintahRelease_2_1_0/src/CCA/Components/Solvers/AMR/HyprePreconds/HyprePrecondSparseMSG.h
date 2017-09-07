/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondSparseMSG
   
   A Hypre SparseMSG (distributed sparse linear solver?) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondSparseMSG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   HypreSolverBase, Precond, HypreSolverBase, HypreSolverParams.

DESCRIPTION 
   Class HyprePrecondSparseMSG sets up and destroys the Hypre
   SparseMSG preconditioner to be used with Hypre solvers. SparseMSG
   is the distributed sparse linear solver. SparseMSG preconditioner
   is often used with CG or GMRES.
  
WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondBase.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondSparseMSG : public HyprePrecondBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondSparseMSG(void) : HyprePrecondBase(initPriority()) {}
    virtual ~HyprePrecondSparseMSG(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondSparseMSG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h
