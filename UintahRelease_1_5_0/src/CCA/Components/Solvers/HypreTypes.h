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

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreTypes_h
#define Packages_Uintah_CCA_Components_Solvers_HypreTypes_h

/*--------------------------------------------------------------------------
CLASS
   HypreTypes
   
   Wrapper of a Hypre solver for a particular variable type.

GENERAL INFORMATION

   File: HypreTypes.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS

DESCRIPTION 

WARNING

--------------------------------------------------------------------------*/

// hypre includes
#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_sstruct_ls.h>
#include <krylov.h>
#include <_hypre_sstruct_mv.h>
#include <_hypre_sstruct_ls.h>

//#define HYPRE_TIMING
#ifndef HYPRE_TIMING
#  ifndef hypre_ClearTiming
     // This isn't in utilities.h for some reason...
#    define hypre_ClearTiming()
#  endif
#endif

// vector must be included after krylov.h because krylov defines 'max'
// and this causes some strange error in something vector includes.
#include <vector>

namespace Uintah {

  //---------- Types ----------
  
  // WARNING WARNING WARNING WARNING
  // Hypre system interface for the solver. If you add or delete items
  // from this list, you need to modify (among other places)
  // HypreSolverBase::assertInterface() and change the rules for
  // requiresPar.
 enum HypreInterface {
    HypreStruct      = 0x1,
    HypreSStruct     = 0x2,
    HypreParCSR      = 0x4,
    HypreInterfaceNA = 0x8
  };

  // Left/right boundary in each dim
  enum BoxSide {
    LeftSide  = -1,
    RightSide = 1,
    BoxSideNA = 3
  };
  
  // Hypre solvers
  enum SolverType {
    SMG, PFMG, SparseMSG, CG, Hybrid, GMRES, AMG, FAC
  };
  
  // Hypre preconditioners
  enum PrecondType {
    PrecondNA, // No preconditioner, use solver directly
    PrecondSMG, PrecondPFMG, PrecondSparseMSG, PrecondJacobi,
    PrecondDiagonal, PrecondAMG, PrecondFAC
  };

  // List of Solver/Preconditioner interface priorities
  typedef std::vector<HypreInterface> Priorities;
    
  
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreTypes_h
