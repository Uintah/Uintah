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

      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      //
      virtual void sched_reComputeTurbSubmodel(const LevelP&, 
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
      virtual void calcVelocityWallBC(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index,
				      int eqnType);

      ///////////////////////////////////////////////////////////////////////
      //
      // Calculate the velocity source terms
      //    [in] 
      //        index = documentation here
      //
      virtual void calcVelocitySource(const ProcessorGroup*,
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
      void computeTurbSubmodel(const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

      ///////////////////////////////////////////////////////////////////////
      //
      // Actually reCalculate the Turbulence sub model
      //    [in] 
      //        documentation here
      //
      void reComputeTurbSubmodel(const ProcessorGroup*,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);

private:

      PhysicalConstants* d_physicalConsts;
      double d_CF; //model constant
      double d_factorMesh; // lengthscale = fac_mesh*meshsize
      double d_filterl; // prescribed filter length scale

      // const VarLabel* variables 
      // for computeTurbulenceSubmodel
      const VarLabel* d_cellInfoLabel;
      const VarLabel* d_uVelocitySPLabel;
      const VarLabel* d_vVelocitySPLabel;
      const VarLabel* d_wVelocitySPLabel;
      const VarLabel* d_densityCPLabel;
      const VarLabel* d_viscosityINLabel;
      const VarLabel* d_cellTypeLabel;
      const VarLabel* d_viscosityCTSLabel;
      const VarLabel* d_uVelocityMSLabel;
      const VarLabel* d_vVelocityMSLabel;
      const VarLabel* d_wVelocityMSLabel;
      const VarLabel* d_densityRCPLabel;
      const VarLabel* d_viscosityRCTSLabel;

      // for calcVelocityWallBC
      const VarLabel* d_uVelocitySIVBCLabel;
      const VarLabel* d_vVelocitySIVBCLabel;
      const VarLabel* d_wVelocitySIVBCLabel;
      const VarLabel* d_uVelocityCPBCLabel;
      const VarLabel* d_vVelocityCPBCLabel;
      const VarLabel* d_wVelocityCPBCLabel;
      const VarLabel* d_densitySIVBCLabel;
      const VarLabel* d_uVelLinSrcPBLMLabel;
      const VarLabel* d_vVelLinSrcPBLMLabel;
      const VarLabel* d_wVelLinSrcPBLMLabel;
      const VarLabel* d_uVelNonLinSrcPBLMLabel;
      const VarLabel* d_vVelNonLinSrcPBLMLabel;
      const VarLabel* d_wVelNonLinSrcPBLMLabel;
      const VarLabel* d_uVelLinSrcMBLMLabel;
      const VarLabel* d_vVelLinSrcMBLMLabel;
      const VarLabel* d_wVelLinSrcMBLMLabel;
      const VarLabel* d_uVelNonLinSrcMBLMLabel;
      const VarLabel* d_vVelNonLinSrcMBLMLabel;
      const VarLabel* d_wVelNonLinSrcMBLMLabel;

}; // End class SmagorinkyModel
  
}  // End namespace ArchesSpace
  
}  // End namespace Uintah

#endif

//
// $Log : $
//

