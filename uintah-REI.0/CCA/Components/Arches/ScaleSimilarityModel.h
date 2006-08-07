//----- ScaleSimilarityModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ScaleSimilarityModel_h
#define Uintah_Component_Arches_ScaleSimilarityModel_h

/**************************************
CLASS
   ScaleSimilarityModel
   
   Class ScaleSimilarityModel is an LES model for
   computing sub-grid scale turbulent viscosity.


GENERAL INFORMATION
   ScaleSimilarityModel.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ScaleSimilarityModel is an LES model for
   computing sub-grid scale turbulent viscosity.


WARNING
   none
****************************************/
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>

namespace Uintah {
class PhysicalConstants;
class BoundaryCondition;

class ScaleSimilarityModel: public TurbulenceModel {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for ScaleSimilarityModel.
      ScaleSimilarityModel(const ArchesLabel* label, 
		       const MPMArchesLabel* MAlb,
		       PhysicalConstants* phyConsts,
		       BoundaryCondition* bndryCondition);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for ScaleSimilarityModel.
      virtual ~ScaleSimilarityModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls,
			         const TimeIntegratorLabel* timelabels);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_computeScalarVariance(SchedulerP&,
					       const PatchSet* patches,
					       const MaterialSet* matls,
			                 const TimeIntegratorLabel* timelabels);

      virtual void sched_computeScalarDissipation(SchedulerP&,
						  const PatchSet* patches,
						  const MaterialSet* matls,
			                 const TimeIntegratorLabel* timelabels);
      ///////////////////////////////////////////////////////////////////////
      // Get the molecular viscosity
      double getMolecularViscosity() const; 

      ///////////////////////////////////////////////////////////////////////
      // Schedule the initialization of the Smagorinsky Coefficient
      //    [in] 
      //        data User data needed for solve 
      
      // GROUP: Access Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the molecular viscosity
//      double getMolecularViscosity() const; 

      ////////////////////////////////////////////////////////////////////////
      // Get the Smagorinsky model constant
      double getSmagorinskyConst() const {
	return d_CF;
      }
      inline void set3dPeriodic(bool periodic) {}
      inline double getTurbulentPrandtlNumber() const {
	return d_turbPrNo;
      }
      inline bool getDynScalarModel() const {
	return false;
      }
      ///////////////////////////////////////////////////////////////////////

protected:
      PhysicalConstants* d_physicalConsts;
      BoundaryCondition* d_boundaryCondition;
      double d_turbPrNo; // turbulent prandtl number
//      bool d_filter_cs_squared; //option for filtering Cs^2 in CompDynamic Procedure
//      bool d_3d_periodic;
      bool d_dynScalarModel;
      double d_lower_limit;
      double d_CL;
      double d_CB;
private:

      // GROUP: Constructors (not instantiated):
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for ScaleSimilarityModel.
      ScaleSimilarityModel();

      // GROUP: Action Methods (private)  :
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

      void computeScalarDissipation(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw,
			            const TimeIntegratorLabel* timelabels);

private:
      double d_CF; //model constant
      // const VarLabel* variables 

}; // End class ScaleSimilarityModel
} // End namespace Uintah
  
  

#endif

// $Log : $

