
#ifndef Uintah_Components_Arches_LinearSolver_h
#define Uintah_Components_Arches_LinearSolver_h

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

class ProcessorGroup;
class LoadBalancer;
class ArchesVariables;
class ArchesConstVariables;
class ArchesLabel;
class CellInformation;
// class StencilMatrix;

using namespace SCIRun;

/**************************************
CLASS
   LinearSolver
   
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   LinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.



WARNING
none
****************************************/

class LinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a LinearSolver.
      // PRECONDITIONS
      // POSTCONDITIONS
      LinearSolver();


      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~LinearSolver();


      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Setup the problem (etc.)
      virtual void problemSetup(const ProblemSpecP& params) = 0;

      inline double getInitNorm() { return init_norm; }

      ////////////////////////////////////////////////////////////////////////
      // Pressure Underrelaxation
      virtual void computePressUnderrelax(const ProcessorGroup* pc,
					  const Patch* patch,
					  ArchesVariables* vars,
				          ArchesConstVariables* constvars) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Pressure Solve
      virtual void pressLisolve(const ProcessorGroup* pc,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw, ArchesVariables* vars, 
				const ArchesLabel* lab) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Calculate pressure residuals
      virtual void computePressResidual(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for pressure equation
      virtual void computePressOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      // Velocity Underrelaxation
      virtual void computeVelUnderrelax(const ProcessorGroup* pc,
					const Patch* patch,
					int index,
					ArchesVariables* vars,
				        ArchesConstVariables* constvars) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Velocity Solve
      virtual void velocityLisolve(const ProcessorGroup* pc,
				   const Patch* patch,
				   int index,
				   double delta_t,
				   ArchesVariables* vars,
				   CellInformation* cellinfo,
				   const ArchesLabel* lab) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Calculate Velocity residuals
      virtual void computeVelResidual(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, int index,
				      ArchesVariables* vars) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for Velocity equation
      virtual void computeVelOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


      ////////////////////////////////////////////////////////////////////////
      // Scalar Underrelaxation
      virtual void computeScalarUnderrelax(const ProcessorGroup* pc,
					   const Patch* patch,
					   int index,
					   ArchesVariables* vars,
				           ArchesConstVariables* constvars) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      virtual void scalarLisolve(const ProcessorGroup* pc,
				 const Patch* patch,
				 int index,
				 double delta_t,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars,
				 CellInformation* cellinfo) = 0;

      virtual void computeEnthalpyUnderrelax(const ProcessorGroup* pc,
					     const Patch* patch,
					     ArchesVariables* vars,
					     ArchesConstVariables* constvars)=0;

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      virtual void enthalpyLisolve(const ProcessorGroup* pc,
				   const Patch* patch,
				   double delta_t,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars,
				   CellInformation* cellinfo) = 0;

       ////////////////////////////////////////////////////////////////////////
      // Calculate Scalar residuals
      virtual void computeScalarResidual(const ProcessorGroup* pc,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw, int index,
					 ArchesVariables* vars) = 0;

      ////////////////////////////////////////////////////////////////////////
      // Calculate order of magnitude term for scalar equation
      virtual void computeScalarOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars) = 0;


   virtual void matrixCreate(const PatchSet* allpatches,
			     const PatchSubset* mypatches) = 0;
   virtual void setPressMatrix(const ProcessorGroup* pc, const Patch* patch,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       const ArchesLabel* lab) = 0;
   

   virtual bool pressLinearSolve() = 0;
   virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars) = 0;
   virtual void destroyMatrix() = 0;
   double init_norm;

protected:

private:

}; // End class LinearSolve

} // End namespace Uintah

#endif  
