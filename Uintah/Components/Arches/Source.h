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
namespace ArchesSpace {
class TurbulenceModel;
 using namespace SCICore::Containers;

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

   Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const);
  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~Source();
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set source terms. Will need more parameters...like velocity and
   // scalars
   void calculatePressureSource(const ProcessorContext* pc,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t, const int index);
   void calculateVelocitySource(const ProcessorContext* pc,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t, const int index);
   void calculateScalarSource(const ProcessorContext* pc,
			      const Region* region,
			      const DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      double delta_t, const int index);
   void modifyVelMassSource(const ProcessorContext* pc,
			    const Region* region,
			    const DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw,
			    double delta_t, const int index);
   void modifyScalarMassSource(const ProcessorContext* pc,
			       const Region* region,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw,
			       double delta_t, const int index);
   void addPressureSource(const ProcessorContext* pc,
			  const Region* region,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
			  const int index);
  
 private:
   TurbulenceModel* d_turbModel;
   PhysicalConstants* d_physicalConsts;

};

}
}
#endif  
  
