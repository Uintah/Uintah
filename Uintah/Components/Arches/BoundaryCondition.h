/**************************************
CLASS
   BoundaryCondition
   
   Class BoundaryCondition applies boundary conditions
   at physical boundaries. For boundary cell types it
   modifies stencil coefficients and source terms.

GENERAL INFORMATION
   BoundaryCondition.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class BoundaryCondition applies boundary conditions
   at physical boundaries. For boundary cell types it
   modifies stencil coefficients and source terms. 



WARNING
none
****************************************/
#ifndef included_BoundaryCondition
#define included_BoundaryCondition
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace ArchesSpace {
class StencilMatrix;
 using namespace SCICore::Containers;

 class TurbulenceModel; 

class BoundaryCondition
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a BoundaryCondition.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   BoundaryCondition();

   BoundaryCondition(TurbulenceModel* d_turb_model);
  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~BoundaryCondition();
   
   void problemSetup(const ProblemSpecP& params);
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set boundary conditions terms. 
   
   void sched_velocityBC(const int index,
			 const LevelP& level,
			 SchedulerP& sched,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void sched_pressureBC(const LevelP& level,
			 const Region* region,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void sched_scalarBC(const int index,
		       const LevelP& level,
		       SchedulerP& sched,
		       const DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw);
   // Set inlet velocity bc's, we need to do it because of staggered grid
   // 
   void sched_setInletVelocityBC(const LevelP& level,
				 SchedulerP& sched,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);
   // used for pressure boundary type
   void sched_computePressureBC(const LevelP& level,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
   // initializes velocities, scalars and properties at the bndry
   // remove sched_setFlat and sched_setProps
   void sched_setProfile(const LevelP& level,
			 SchedulerP& sched,
			 const DataWarehouseP& old_dw);

   // assign flat velocity profiles at the inlet
   void sched_setFlatProfile(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw);
   void sched_setPropsProfile(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw);
 private:
   void setFlatProfile(const ProcessorContext* pc,
		       const Region* region,
		       const DataWarehouseP& old_dw);
   void setPropsProfile(const ProcessorContext* pc,
		       const Region* region,
		       const DataWarehouseP& old_dw);
   void setInletVelocityBC(const ProcessorContext* pc,
			   const Region* region,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
   void calculatePressBC(const ProcessorContext* pc,
			 const Region* region,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void velocityBC(const ProcessorContext* pc,
		   const Region* region,
		   const DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw,
		   const int index);
   void pressureBC(const ProcessorContext*,
		   const Region* region,
		   const DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw);
   void scalarBC(const ProcessorContext* pc,
		 const Region* region,
		 const DataWarehouseP& old_dw,
		 DataWarehouseP& new_dw,
		 const int index);
   // used for calculating wall boundary conditions
   TurbulenceModel* d_turbModel;
   // Diff BC types
#if 0
   struct FlowInlet {
     // define enum for cell type
     CellTypeInfo inletType; 
     // input vars
     double flowRate;
     // array of size numMixingVars -1
     Array1<double> streamMixturefraction;
     double turb_lengthScale;
     // calculated values
     double density;
     // inlet area
     double area;
     // need a constructor
     FlowInlet(int numMix);
     problemSetup(ProblemSpecP& params);
   };
   struct PressureInlet {
     CellTypeInfo pressureType;
     // array of size numMixingVars -1
     Array1<double> streamMixturefraction;
     double turb_lengthScale;
     double density;
     double refPressure;
     PressureInlet(int numMix);
     problemSetup(ProblemSpecP& params);
   };
   struct FlowOutlet {
     CellTypeInfo outletType;
     // imp for convergence
     Array1<double> streamMixturefraction;
     double turb_lengthScale;
     double density;
     double area;
     FlowOutlet(int numMix);
     problemSetup(ProblemSpecP& params);
   };
#endif
};
}
}
#endif  
  
