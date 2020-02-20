/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef Uintah_Components_Arches_RadHypreSolver_h
#define Uintah_Components_Arches_RadHypreSolver_h

#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>


#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>


namespace Uintah {

class ProcessorGroup;

/**************************************
CLASS
   RadHypreSolver
   
   Class RadHypreSolver uses gmres solver
   solver

GENERAL INFORMATION
   RadHypreSolver.h - declaration of the class
   
   Author: Gautham Krishnamoorthy (gautham@crsim.utah.edu)
   
   Creation Date:   June 30, 2004
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class RadHypreSolver is a linear solver with multigrid

WARNING
   none

****************************************/

class RadHypreSolver: public RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a HypreSolver.
      RadHypreSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RadHypreSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      ////////////////////////////////////////////////////////////////////////
      // HYPRE grid and stencil setup
      void gridSetup(bool plusX, 
                     bool plusY, 
                     bool plusZ);

      ////////////////////////////////////////////////////////////////////////
       // to close petsc
      void finalizeSolver();

      void matrixCreate(const PatchSet* allpatches,
                            const PatchSubset* mypatc) {};

      void matrixInit(const Patch* patch) ;


      void setMatrix(const ProcessorGroup* pc,
                     const Patch* patch,
                     ArchesVariables* vars,
                     ArchesConstVariables* constvars, 
                     bool plusX, 
                     bool plusY,
                     bool plusZ,
                     CCVariable<double>& SU,
                     CCVariable<double>& AB,
                     CCVariable<double>& AS,
                     CCVariable<double>& AW,
                     CCVariable<double>& AP,
                     const bool print );

      bool radLinearSolve( const int dir, const bool print_all_info );

      virtual void copyRadSoln(const Patch* patch, 
                               ArchesVariables* vars);
      virtual void destroyMatrix();
protected:

private:
  std::string d_pcType;
  std::string d_kspType;
  std::string d_kspFix;
  int d_overlap;
  int d_fill;
  int d_maxSweeps;
  int **d_iupper, **d_ilower, **d_offsets;
  int d_volume, d_nblocks, d_dim, d_stencilSize;
  int *d_stencilIndices;
  int d_A_num_ghost[6];
  int d_iteration;
  bool d_use7PointStencil;    ///> Tells the Arches-Hypre solver to setup a 7 point stencil instead of the default 4 point stencil

  double d_convgTol; // convergence tolerence
  double d_initResid;
  double d_residual;
  double d_stored_residual;
  double init_norm;
  
  double *d_valueA;
  double *d_valueB;
  double *d_valueX;
  const ProcessorGroup* d_myworld;
  std::map<const Patch*, int> d_petscGlobalStart;
  std::map<const Patch*, Array3<int> > d_petscLocalToGlobal;
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;

}; // End class RadHypreSolver.h

} // End namespace Uintah

#endif  




