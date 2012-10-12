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

//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Models/Radiation/Models_PetscSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;

// ****************************************************************************
// Default constructor for Models_PetscSolver
// ****************************************************************************
Models_PetscSolver::Models_PetscSolver(const ProcessorGroup* myworld) :
  d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Models_PetscSolver::~Models_PetscSolver()
{
// Comment out the following till better place to finalize PETSC is found
//  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
Models_PetscSolver::problemSetup(const ProblemSpecP& /*params*/, 
                                 bool /* shradiation */)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
Models_PetscSolver::outputProblemSpec(ProblemSpecP& /*params*/)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
Models_PetscSolver::matrixCreate(const PatchSet* ,
                                 const PatchSubset* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
Models_PetscSolver::setMatrix(const ProcessorGroup* ,
                           const Patch* ,
                           RadiationVariables* ,
                           bool , bool , bool ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& ,
                           CCVariable<double>& )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


bool
Models_PetscSolver::radLinearSolve()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void
Models_PetscSolver::copyRadSoln(const Patch* , RadiationVariables* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

void
Models_PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


// Shutdown PETSc
void
Models_PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

