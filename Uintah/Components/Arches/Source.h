/**************************************
CLASS
   Source
   
   Class Source computes source terms for 
   N-S equations.  

GENERAL INFORMATION
   Source.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class Source computes source terms for 
   N-S equations.  



WARNING
none
****************************************/
#ifndef included_Source
#define included_Source
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace Components {
class TurbulenceModel;
 using namespace Uintah::Grid;
 using namespace Uintah::Interface;
 using namespace SCICore::Containers;
 using namespace Uintah::Parallel;

class Source : 
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a Source.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   Source();

   Source(TurbulenceModel* turb_model);
  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~Source();
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set source terms. Will need more parameters...like velocity and
   // scalars
   
   void sched_calculatePressureSource(const LevelP& level,
				      SchedulerP& sched,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);
   void sched_calculateVelocitySource(int index,const LevelP& level,
				      SchedulerP& sched,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);
   void sched_calculateScalarSource(int index,const LevelP& level,
				    SchedulerP& sched,
				    const DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);
 private:
   void calculatePressureSource(const Region* region,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
   void calculateVelocitySource(int index,const Region* region,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
   void calculateScalarSource(int index,const Region* region,
			      SchedulerP& sched,
			      const DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);
   TurbulenceModel* d_turbModel;

};

}
}
#endif  
  
