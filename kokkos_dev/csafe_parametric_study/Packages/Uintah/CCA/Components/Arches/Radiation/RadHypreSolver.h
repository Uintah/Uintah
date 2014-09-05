#ifndef Uintah_Components_Arches_RadHypreSolver_h
#define Uintah_Components_Arches_RadHypreSolver_h

#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>


namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

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
   
   Copyright U of U 2000

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
      void gridSetup(const ProcessorGroup*,
		     const Patch* patch, bool plusX, bool plusY, bool plusZ);

      ////////////////////////////////////////////////////////////////////////
       // to close petsc
      void finalizeSolver();

      virtual void matrixCreate(const PatchSet* allpatches,
			    const PatchSubset* mypatc) {};

      void setMatrix(const ProcessorGroup* pc,
			    const Patch* patch,
			    ArchesVariables* vars,
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

      virtual void copyRadSoln(const Patch* patch, ArchesVariables* vars);
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
  double d_initResid;
  double d_residual;
  double d_stored_residual;
  double init_norm;

  double *d_value;
  const ProcessorGroup* d_myworld;
  map<const Patch*, int> d_petscGlobalStart;
  map<const Patch*, Array3<int> > d_petscLocalToGlobal;
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;

}; // End class RadHypreSolver.h

} // End namespace Uintah

#endif  




