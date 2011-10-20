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


//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/PetscSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;

// ****************************************************************************
// Default constructor for PetscSolver
// ****************************************************************************
PetscSolver::PetscSolver(const ProcessorGroup* myworld) :
  d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PetscSolver::~PetscSolver()
{
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
PetscSolver::problemSetup(const ProblemSpecP&)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
PetscSolver::matrixCreate(const PatchSet*,
                          const PatchSubset*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
PetscSolver::setMatrix(const ProcessorGroup* ,
                       const Patch*,
                       CCVariable<Stencil7>& coeff )
{
  throw InternalError("setMatrix(): PetscSolver not configured", __FILE__, __LINE__);
}

void
PetscSolver::setRHS_X( const ProcessorGroup * pc, 
                       const Patch          * patch,
                       constCCVariable<double>& guess,
                       constCCVariable<double>& rhs,
                       bool construct_A )
{
  throw InternalError("set RHS_X(): PetscSolver not configured", __FILE__, __LINE__);
}

bool
PetscSolver::pressLinearSolve()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
  return false;
}


void
PetscSolver::copyPressSoln(const Patch*, ArchesVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}
  
void
PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

void
PetscSolver::print(const string& desc, const int timestep, const int step)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}
