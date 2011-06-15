/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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



#ifndef Uintah_Components_Arches_PetscSolver_h
#define Uintah_Components_Arches_PetscSolver_h

#include <sci_defs/petsc_defs.h>

#include <CCA/Components/Arches/LinearSolver.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <Core/Grid/Patch.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

namespace Uintah {

class ProcessorGroup;

/**************************************
CLASS
   PetscSolver
   
   Class PetscSolver uses gmres solver
   solver

GENERAL INFORMATION
   PetscSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class PetscSolver is a gmres linear solver

WARNING
   none

****************************************/

class PetscSolver: public LinearSolver {

public:
  PetscSolver(const ProcessorGroup* myworld);

  virtual ~PetscSolver();

  void problemSetup(const ProblemSpecP& params);

  virtual void matrixCreate(const PatchSet* allpatches,
                            const PatchSubset* mypatches);
                            
  virtual void setMatrix(const ProcessorGroup* pc, 
                         const Patch* patch,
                         constCCVariable<Stencil7>& coeff);

  virtual void setRHS_X(const ProcessorGroup* pc, 
                        const Patch* patch,
                        CCVariable<double>& guess,
                        constCCVariable<double>& rhs, 
                        bool construct_A );


  virtual bool pressLinearSolve();
  
  virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars);
 
  virtual void destroyMatrix();
protected:

private:
  string d_pcType;
  string d_solverType;
  int d_overlap;
  int d_fill;
  int d_maxSweeps;
  double d_residual;
  const ProcessorGroup* d_myworld;
  
  
#ifdef HAVE_PETSC
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b, d_u;
#endif
}; // End class PetscSolver.h

} // End namespace Uintah

#endif  
  
