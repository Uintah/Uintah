
#ifndef Uintah_Components_Arches_RBGSSolver_h
#define Uintah_Components_Arches_RBGSSolver_h

/**************************************
CLASS
   RBGSSolver
   
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver

GENERAL INFORMATION
   RBGSSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver

WARNING
   none

****************************************/

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

using namespace SCIRun;

class RBGSSolver: public LinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a RBGSSolver.
      RBGSSolver();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RBGSSolver();

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

      ////////////////////////////////////////////////////////////////////////
      // Scalar Underrelaxation
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
   virtual void matrixCreate(const PatchSet* allpatches,
			     const PatchSubset* mypatches);
   virtual void setPressMatrix(const ProcessorGroup* pc, const Patch* patch,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       const ArchesLabel* lab);
   virtual bool pressLinearSolve();
   virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars);
   virtual void destroyMatrix();

protected:

private:

      int d_maxSweeps;
      double d_convgTol; // convergence tolerence
      double d_underrelax;
      double d_initResid;
      double d_residual;

}; // End class RBGSSolver.h

} // End namespace Uintah

#endif  
  
