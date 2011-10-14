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


//----- HypreSolver.cc ----------------------------------------------

#include <fstream> // work around compiler bug with RHEL 3

#include <CCA/Components/Arches/HypreSolverV2.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "krylov.h"
#include "_hypre_struct_mv.h"

using namespace std;
using namespace Uintah;


// ****************************************************************************
// Default constructor for HypreSolver
// ****************************************************************************
HypreSolver::HypreSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
HypreSolver::~HypreSolver()
{
}

//______________________________________________________________________
//
void 
HypreSolver::problemSetup(const ProblemSpecP& params)
{
}


//______________________________________________________________________
//
void 
HypreSolver::setMatrix(const ProcessorGroup* pc,
                       const Patch* patch,
                       CCVariable<Stencil7>& coeff)
{ 

}
//______________________________________________________________________
//
void 
HypreSolver::setRHS_X(const ProcessorGroup* pc,
                      const Patch* patch,
                      constCCVariable<double>& guess,
                      constCCVariable<double>& rhs, 
                      bool construct_A )
{ 
}
//______________________________________________________________________
//
bool
HypreSolver::pressLinearSolve()
{
}
//______________________________________________________________________
// 
void
HypreSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
}
 
//______________________________________________________________________
//  
void
HypreSolver::destroyMatrix() 
{
}

//______________________________________________________________________
//
void HypreSolver::print(const string& desc, const int timestep, const int step){
#if 0
  char A_fname[100],B_fname[100], X_fname[100];
  
  sprintf(B_fname,"output/b.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(X_fname,"output/x.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(A_fname,"output/A.%s.%i.%i",desc.c_str(), timestep, step);
  
  HYPRE_StructMatrixPrint(A_fname, d_A, 0);
  
  HYPRE_StructVectorPrint(B_fname, d_b, step);
  
  HYPRE_StructVectorPrint(X_fname, d_x, step); 
#endif 
}
