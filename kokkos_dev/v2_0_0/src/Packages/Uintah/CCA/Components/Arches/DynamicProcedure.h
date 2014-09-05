//----- DynamicProcedure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_DynamicProcedure_h
#define Uintah_Component_Arches_DynamicProcedure_h

/**************************************
CLASS
   DynamicProcedure
   
   Class DynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


GENERAL INFORMATION
   DynamicProcedure.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class DynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


WARNING
   none
****************************************/

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>

namespace Uintah {
class PhysicalConstants;
class BoundaryCondition;


class DynamicProcedure: public SmagorinskyModel {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for DynamicProcedure.
      DynamicProcedure(const ArchesLabel* label, 
		       const MPMArchesLabel* MAlb,
		       PhysicalConstants* phyConsts,
		       BoundaryCondition* bndryCondition);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for DynamicProcedure.
      virtual ~DynamicProcedure();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 

      virtual void sched_computeTurbSubmodel(const LevelP&, 
					     SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls,
			                 const TimeIntegratorLabel* timelabels);

      virtual void sched_computeScalarVariance(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls,
			    		const TimeIntegratorLabel* timelabels);


protected:

private:

      // GROUP: Constructors (not instantiated):
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for DynamicProcedure.
      DynamicProcedure();

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
				 DataWarehouse* new_dw,
			         const TimeIntegratorLabel* timelabels);
 
      void reComputeFilterValues(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
			         const TimeIntegratorLabel* timelabels);
 
      void reComputeSmagCoeff(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      const TimeIntegratorLabel* timelabels);


      ///////////////////////////////////////////////////////////////////////
      // Actually Calculate the subgrid scale variance
      //    [in] 
      //        documentation here
      void computeScalarVariance(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
			         const TimeIntegratorLabel* timelabels);


 private:

      // const VarLabel* variables 

 }; // End class DynamicProcedure
} // End namespace Uintah
  
  

#endif

// $Log : $

