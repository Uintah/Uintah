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
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace Components {
class StencilMatrix;
 using namespace Uintah::Grid;
 using namespace Uintah::Interface;
 using namespace SCICore::Containers;
 using namespace Uintah::Parallel;

 class TurbulenceModel; 

class BoundaryCondition : 
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
   // assign flat velocity profiles at the inlet
   void sched_setFlatProfile(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);
 private:
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
   TurbulenceModel* d_turb_model;



};
}
}
#endif  
  
