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

#ifndef Uintah_Components_Models_Models_HypreSolver_h
#define Uintah_Components_Models_Models_HypreSolver_h

#include <CCA/Components/Models/Radiation/Models_RadiationSolver.h>
#include <CCA/Components/Models/Radiation/RadiationVariables.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>

#include <iostream>

namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

/**************************************
CLASS
   Models_HypreSolver
   
   Class Models_HypreSolver uses gmres solver
   solver

GENERAL INFORMATION
   Models_HypreSolver.h - declaration of the class
   
   Author: Gautham Krishnamoorthy (gautham@crsim.utah.edu)
   
   Creation Date:   June 30, 2004
   
   C-SAFE 
   
   
KEYWORDS


DESCRIPTION
   Class Models_HypreSolver is a linear solver with multigrid

WARNING
   none

****************************************/

class Models_HypreSolver: public Models_RadiationSolver {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a HypreSolver.
  Models_HypreSolver(const ProcessorGroup* myworld);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
  virtual ~Models_HypreSolver();

  // GROUP: Problem Setup:
  ////////////////////////////////////////////////////////////////////////
  // Problem setup
  void problemSetup(const ProblemSpecP& params, bool shradiation);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  ////////////////////////////////////////////////////////////////////////
  // to close hypre
  void finalizeSolver();

  ////////////////////////////////////////////////////////////////////////
  // HYPRE grid and stencil setup
  void gridSetup(const ProcessorGroup*,
                 const Patch* patch, bool plusX, bool plusY, bool plusZ);

  virtual void matrixCreate(const PatchSet* /*allpatches*/,
                            const PatchSubset* /*mypatc*/) {
//    std::cout << "WARNING: Models_HypreSolver.h: matrixCreate NOT IMPLEMENTED!!!"<< endl;
  };

  void setMatrix(const ProcessorGroup* pc,
                 const Patch* patch,
                 RadiationVariables* vars,
                 bool plusX, bool plusY, bool plusZ,
                 CCVariable<double>& SU,
                 CCVariable<double>& AB,
                 CCVariable<double>& AS,
                 CCVariable<double>& AW,
                 CCVariable<double>& AP,
                 CCVariable<double>& AE,
                 CCVariable<double>& AN,
                 CCVariable<double>& AT);

  bool radLinearSolve();

  virtual void copyRadSoln(const Patch* patch, RadiationVariables* vars);
  virtual void destroyMatrix();
protected:

private:
  string d_precondType;
  string d_solverType;
  int d_maxIter;
  double d_tolerance;
  bool d_shrad;

  int **d_iupper, **d_ilower, **d_offsets;
  int d_volume, d_nblocks, d_dim, d_stencilSize;
  int *d_stencilIndices;
  int d_A_num_ghost[6];
  double *d_value;
  const ProcessorGroup* d_myworld;
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;

}; // End class Models_HypreSolver.h

} // End namespace Uintah

#endif  
