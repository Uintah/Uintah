//----- TurbulenceModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TurbulenceModel_h
#define Uintah_Component_Arches_TurbulenceModel_h

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

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif
namespace Uintah {
class TimeIntegratorLabel;
class TurbulenceModel
{
public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for TurbulenceModel.
      TurbulenceModel(const ArchesLabel* label, 
		      const MPMArchesLabel* MAlb);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for TurbulenceModel.
      virtual ~TurbulenceModel();
#ifdef PetscFilter
      inline void setFilter(Filter* filter) {
	d_filter = filter;
      }
#endif
      // GROUP: Access Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the molecular viscisity
      virtual double getMolecularViscosity() const = 0;

      ////////////////////////////////////////////////////////////////////////
      // Get the Smagorinsky model constant
      virtual double getSmagorinskyConst() const = 0;

       // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db) = 0;


      // access function
#ifdef PetscFilter
      Filter* getFilter() const{
	return d_filter;
      }
      virtual void set3dPeriodic(bool periodic) = 0;

      void sched_initFilterMatrix(const LevelP&, 
			      SchedulerP&, const PatchSet* patches,
			      const MaterialSet* matls);

#endif
      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_computeTurbSubmodel(const LevelP&, 
					     SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls) = 0;


      ///////////////////////////////////////////////////////////////////////
      // Interface for Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls,
			    	 const TimeIntegratorLabel* timelabels) = 0;


      virtual void sched_computeScalarVariance(SchedulerP&,
					       const PatchSet* patches,
					       const MaterialSet* matls,
			    	     const TimeIntegratorLabel* timelabels) = 0;
      virtual void sched_computeScalarDissipation(SchedulerP&,
						  const PatchSet* patches,
						  const MaterialSet* matls,
			    	     const TimeIntegratorLabel* timelabels) = 0;
 protected:

      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

#ifdef PetscFilter
      Filter* d_filter;
#endif
private:

#ifdef PetscFilter
      void initFilterMatrix(const ProcessorGroup* pg,
			    const PatchSubset* patches,
			    const MaterialSubset*,
			    DataWarehouse*,
			    DataWarehouse* new_dw);
#endif


}; // End class TurbulenceModel
} // End namespace Uintah
  
  

#endif

// $Log :$



