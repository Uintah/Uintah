/**************************************
CLASS
   SmagorinskyModel
   
   Class SmagorinskyModel is an LES model for
   computing sub-grid scale turbulent viscosity.


GENERAL INFORMATION
   SmagorinskyModel.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class SmagorinskyModel is an LES model for
   computing sub-grid scale turbulent viscosity.


WARNING
   none
****************************************/

#ifndef Uintah_Component_Arches_SmagorinskyModel_h
#define Uintah_Component_Arches_SmagorinskyModel_h

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>

namespace Uintah {
  namespace ArchesSpace {

class PhysicalConstants;


class SmagorinskyModel:
public TurbulenceModel
{
public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Blank constructor for SmagorinskyModel.
  SmagorinskyModel(PhysicalConstants* phyConsts);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for SmagorinskyModel.
  virtual ~SmagorinskyModel();

  virtual void problemSetup(const ProblemSpecP& db);
  ////////////////////////////////////////////////////////////////////////
  //    [in] data User data needed for solve 
  virtual void sched_computeTurbSubmodel(const LevelP&, SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw);
  virtual void calcVelocityWallBC(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, int index);
  virtual void calcVelocitySource(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw, int index);


private:
  void computeTurbSubmodel(const ProcessorContext*,
			   const Region* region,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
  PhysicalConstants* d_physicalConsts;
  double d_CF; //model constant
  double d_factorMesh; // lengthscale = fac_mesh*meshsize
  double d_filterl; // prescribed filter length scale
};
  
  } 
  
}

#endif

