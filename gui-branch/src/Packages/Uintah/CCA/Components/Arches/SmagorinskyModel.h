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

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>

namespace Uintah {
class PhysicalConstants;


class SmagorinskyModel: public TurbulenceModel {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for SmagorinskyModel.
      SmagorinskyModel(const ArchesLabel* label, 
		       const MPMArchesLabel* MAlb,
		       PhysicalConstants* phyConsts);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for SmagorinskyModel.
      virtual ~SmagorinskyModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_computeTurbSubmodel(SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_computeScalarVariance(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls);
      // GROUP: Access Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the molecular viscosity
      double getMolecularViscosity() const; 

      ////////////////////////////////////////////////////////////////////////
      // Get the Smagorinsky model constant
      double getSmagorinskyConst() const {
	return d_CF;
      }
       // GROUP: Action Computations :
      ///////////////////////////////////////////////////////////////////////
      // Calculate the wall velocity boundary conditions
      //    [in] 
      //        index = documentation here
      virtual void calcVelocityWallBC(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index,
				      int eqnType);

      ///////////////////////////////////////////////////////////////////////
      // Calculate the velocity source terms
      //    [in] 
      //        index = documentation here
      virtual void calcVelocitySource(const ProcessorGroup*,
				      const Patch* patch,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index);

protected:

private:

      // GROUP: Constructors (not instantiated):
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for SmagorinskyModel.
      SmagorinskyModel();

      // GROUP: Action Methods (private)  :
      ///////////////////////////////////////////////////////////////////////
      // Actually Calculate the Turbulence sub model
      //    [in] 
      //        documentation here
      void computeTurbSubmodel(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Actually reCalculate the Turbulence sub model
      //    [in] 
      //        documentation here
      void reComputeTurbSubmodel(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);
      ///////////////////////////////////////////////////////////////////////
      // Actually Calculate the subgrid scale variance
      //    [in] 
      //        documentation here
      void computeScalarVariance(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);


private:

      PhysicalConstants* d_physicalConsts;
      double d_CF; //model constant
      double d_factorMesh; // lengthscale = fac_mesh*meshsize
      double d_filterl; // prescribed filter length scale
      double d_CFVar; // model constant for mixture fraction variance
      // const VarLabel* variables 
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

}; // End class SmagorinkyModel
} // End namespace Uintah
  
  

#endif

// $Log : $

