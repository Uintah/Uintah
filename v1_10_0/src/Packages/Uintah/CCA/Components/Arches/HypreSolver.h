
#ifndef Uintah_Components_Arches_hypreSolver_h
#define Uintah_Components_Arches_hypreSolver_h

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>

#include <Core/Containers/Array1.h>

#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>


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

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a HypreSolver.
      HypreSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~HypreSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      ////////////////////////////////////////////////////////////////////////
      // Pressure Underrelaxation
      void computePressUnderrelax(const ProcessorGroup* pc,
				  const Patch* patch,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // HYPRE grid and stencil setup
      void gridSetup(const ProcessorGroup*,
		     const Patch* patch);

      ////////////////////////////////////////////////////////////////////////
      // Pressure Solve
      void pressLisolve(const ProcessorGroup* pc,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw, 
			ArchesVariables* vars, 
			const ArchesLabel* lab);

      ////////////////////////////////////////////////////////////////////////
      // Calculate pressure residuals
      void computePressResidual(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);


      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for pressure equation
      void computePressOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);


      ////////////////////////////////////////////////////////////////////////
      // Velocity Underrelaxation
      void computeVelUnderrelax(const ProcessorGroup* pc,
				const Patch* patch,
				int index,
				ArchesVariables* vars,
				ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Velocity Solve
      void velocityLisolve(const ProcessorGroup* pc,
			   const Patch* patch,
			   int index,
			   double delta_t,
			   ArchesVariables* vars,
			   CellInformation* cellinfo,
			   const ArchesLabel* lab);

      ////////////////////////////////////////////////////////////////////////
      // Calculate Velocity residuals
      void computeVelResidual(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, int index,
				      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for Velocity equation
      void computeVelOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Calculate scalar residuals
      void computeScalarResidual(const ProcessorGroup* pc,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw, 
				 int index,
				 ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for scalar equation
      void computeScalarOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Scalar Underrelaxation
      void computeScalarUnderrelax(const ProcessorGroup* pc,
				   const Patch* patch,
				   int index,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      void scalarLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 int index, double delta_t,
			 ArchesVariables* vars,
			 ArchesConstVariables* constvars,
			 CellInformation* cellinfo);
      void computeEnthalpyUnderrelax(const ProcessorGroup* pc,
				     const Patch* patch,
				     ArchesVariables* vars,
				     ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      void enthalpyLisolve(const ProcessorGroup* pc,
			   const Patch* patch,
			   double delta_t,
			   ArchesVariables* vars,
			   ArchesConstVariables* constvars,
			   CellInformation* cellinfo);
       // to close petsc 
      void finalizeSolver();

  virtual void matrixCreate(const PatchSet* allpatches,
			    const PatchSubset* mypatc) {};
  virtual void setPressMatrix(const ProcessorGroup* pc, const Patch* patch,
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
  double *d_value;
  const ProcessorGroup* d_myworld;
  map<const Patch*, int> d_petscGlobalStart;
  map<const Patch*, Array3<int> > d_petscLocalToGlobal;
  HYPRE_StructMatrix d_A;
  HYPRE_StructVector d_x, d_b;
  HYPRE_StructGrid d_grid;
  HYPRE_StructStencil d_stencil;
}; // End class hypreSolver.h

} // End namespace Uintah

#endif  
  
