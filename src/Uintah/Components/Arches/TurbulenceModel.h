/**************************************
CLASS
   TurbulenceModel
   
   Class TurbulenceModel is an abstract base class
   which defines the operations needed to compute
   unresolved turbulence submodels

GENERAL INFORMATION
   TurbulenceModel.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class TurbulenceModel is an abstract base class
   which defines the operations needed to compute
   unresolved turbulence submodels

WARNING
   none
****************************************/

#ifndef Uintah_Component_Arches_TurbulenceModel_h
#define Uintah_Component_Arches_TurbulenceModel_h

#include <Uintah/Components/Arches/Arches.h>

namespace Uintah {
  namespace Components {

#ifndef LACKS_NAMESPACE
using namespace Uintah::Components;
#endif


class TurbulenceModel
{
public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Blank constructor for TurbulenceModel.
  TurbulenceModel();

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for TurbulenceModel.
  virtual ~TurbulenceModel();


  ////////////////////////////////////////////////////////////////////////
  //    [in] data User data needed for solve 
  virtual void sched_computeTurbSubmodel(const LevelP&, SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) = 0;
  virtual void calcVelocityWallBC(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, int index) = 0;
  virtual void calcVelocitySource(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, int index) = 0;
  virtual void problemSetup(const ProblemSpecP& db) = 0;

private:

};
  
  } 
  
}

#endif

