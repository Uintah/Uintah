//----- SmagorinskyModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_SmagorinskyModel_h
#define Uintah_Component_Arches_SmagorinskyModel_h

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

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>

namespace Uintah {
namespace ArchesSpace {

class PhysicalConstants;


class SmagorinskyModel: public TurbulenceModel {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Blank constructor for SmagorinskyModel.
      //
      SmagorinskyModel(PhysicalConstants* phyConsts);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for SmagorinskyModel.
      //
      virtual ~SmagorinskyModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      //
      virtual void sched_computeTurbSubmodel(const LevelP&, 
					     SchedulerP& sched,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw);

      // GROUP: Action Computations :
      ///////////////////////////////////////////////////////////////////////
      //
      // Calculate the wall velocity boundary conditions
      //    [in] 
      //        index = documentation here
      //
      virtual void calcVelocityWallBC(const ProcessorContext*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index);
      ///////////////////////////////////////////////////////////////////////
      //
      // Calculate the velocity source terms
      //    [in] 
      //        index = documentation here
      //
      virtual void calcVelocitySource(const ProcessorContext*,
				      const Patch* patch,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index);

protected:

private:

      // GROUP: Constructors (not instantiated):
      ////////////////////////////////////////////////////////////////////////
      //
      // Blank constructor for SmagorinskyModel.
      //
      SmagorinskyModel();

      // GROUP: Action Methods (private)  :
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually Calculate the Turbulence sub model
      //    [in] 
      //        documentation here
      //
      void computeTurbSubmodel(const ProcessorContext*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

private:

      PhysicalConstants* d_physicalConsts;
      double d_CF; //model constant
      double d_factorMesh; // lengthscale = fac_mesh*meshsize
      double d_filterl; // prescribed filter length scale

      // const VarLabel* variables
      const VarLabel* d_uVelocityLabel;
      const VarLabel* d_vVelocityLabel;
      const VarLabel* d_wVelocityLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;
      const VarLabel* d_cellTypeLabel;
      const VarLabel* d_uLinSrcLabel;
      const VarLabel* d_vLinSrcLabel;
      const VarLabel* d_wLinSrcLabel;
      const VarLabel* d_uNonLinSrcLabel;
      const VarLabel* d_vNonLinSrcLabel;
      const VarLabel* d_wNonLinSrcLabel;

}; // End class SmagorinkyModel
  
}  // End namespace ArchesSpace
  
}  // End namespace Uintah

#endif

//
// $Log : $
//

