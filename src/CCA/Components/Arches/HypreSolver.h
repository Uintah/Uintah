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



#ifndef Uintah_Components_Arches_hypreSolver_h
#define Uintah_Components_Arches_hypreSolver_h

#include <CCA/Components/Arches/LinearSolver.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <Core/Grid/Patch.h>

#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>


namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

/**************************************
CLASS
   hypreSolver
   
   Class hypreSolver uses cg solver
   solver

GENERAL INFORMATION
   hypreSolver.h - declaration of the class
   
   Author: Wing Yee (Wing@crsim.utah.edu)
   
   Creation Date:   May 15, 2002
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class hypreSolver is a linear solver with multigrid

WARNING
   none

****************************************/

class HypreSolver: public LinearSolver {

public:

  HypreSolver(const ProcessorGroup* myworld);

  virtual ~HypreSolver();


  void problemSetup(const ProblemSpecP& params);

  void gridSetup(const ProcessorGroup*,
                 const Patch* patch);
                 
  void finalizeSolver();

  virtual void matrixCreate(const PatchSet* allpatches,
                            const PatchSubset* mypatc) {};
                            
  virtual void setPressMatrix(const ProcessorGroup* pc, 
                              const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars,
                              const ArchesLabel* lab);

  virtual void setPressRHS(const ProcessorGroup* pc, 
                           const Patch* patch,
                           ArchesVariables* vars,
                           ArchesConstVariables* constvars,
                           const ArchesLabel* lab);

  virtual bool pressLinearSolve();
  virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars);
  virtual void destroyMatrix();
protected:

private:
  string d_solverType;

  int d_maxSweeps;
  int **d_iupper, **d_ilower, **d_offsets;
  int d_volume, d_nblocks, d_dim, d_stencilSize;
  int *d_stencilIndices;
  int d_A_num_ghost[6];
  
  double d_residual;
  double d_stored_residual;
  const ProcessorGroup* d_myworld;
  
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;
}; // End class hypreSolver.h

} // End namespace Uintah

#endif  
  
