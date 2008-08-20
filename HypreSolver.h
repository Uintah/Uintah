
#ifndef Uintah_Components_Arches_hypreSolver_h
#define Uintah_Components_Arches_hypreSolver_h

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

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
   

  virtual bool pressLinearSolve();
  virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars);
  virtual void destroyMatrix();
protected:

private:
  string d_pcType;
  string d_kspType;

  int d_maxSweeps;
  int **d_iupper, **d_ilower, **d_offsets;
  int d_volume, d_nblocks, d_dim, d_stencilSize;
  int *d_stencilIndices;
  int d_A_num_ghost[6];
  
  double d_residual;
  double d_stored_residual;
  double *d_value;
  const ProcessorGroup* d_myworld;
  
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;
}; // End class hypreSolver.h

} // End namespace Uintah

#endif  
  
