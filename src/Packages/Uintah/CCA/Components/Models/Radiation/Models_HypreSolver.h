#ifndef Uintah_Components_Models_Models_HypreSolver_h
#define Uintah_Components_Models_Models_HypreSolver_h

#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

#include <utilities.h>
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
   
   Copyright U of U 2000

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

  ////////////////////////////////////////////////////////////////////////
  // HYPRE grid and stencil setup
  void gridSetup(const ProcessorGroup*,
                 const Patch* patch, bool plusX, bool plusY, bool plusZ);

  ////////////////////////////////////////////////////////////////////////
  // to close petsc
  void finalizeSolver();

  virtual void matrixCreate(const PatchSet* /*allpatches*/,
                            const PatchSubset* /*mypatc*/) {
    std::cout << "WARNING: Models_HypreSolver.h: matrixCreate NOT IMPLEMENTED!!!"<< endl;
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
  string d_pcType;
  string d_kspType;
  string d_kspFix;
  int d_overlap;
  int d_fill;
  int d_maxSweeps;
  int **d_iupper, **d_ilower, **d_offsets;
  int d_volume, d_nblocks, d_dim, d_stencilSize;
  int *d_stencilIndices;
  int d_A_num_ghost[6];
  double d_convgTol; // convergence tolerence
  double d_underrelax;
  double d_initResid;
  double d_residual;
  double init_norm;

  bool d_shrad;

  double *d_value;
  const ProcessorGroup* d_myworld;
  map<const Patch*, int> d_petscGlobalStart;
  map<const Patch*, Array3<int> > d_petscLocalToGlobal;
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;

}; // End class Models_HypreSolver.h

} // End namespace Uintah

#endif  
