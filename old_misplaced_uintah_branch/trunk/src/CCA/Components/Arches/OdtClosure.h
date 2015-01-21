//----- OdtClosure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_OdtClosure_h
#define Uintah_Component_Arches_OdtClosure_h

/**************************************
CLASS
   OdtClosure
   
   Class OdtClosure is an LES model for
   computing sub-grid scale turbulent stress.


GENERAL INFORMATION
   OdtClosure.h - declaration of the class
   
   Author: Zhaosheng Gao (zgao@crsim.utah.edu)
      
   Creation Date:   November 12, 2004
   
   C-SAFE 
   
   Copyright U of U 2004

KEYWORDS


DESCRIPTION
   Class OdtClosure is an LES model for
   computing sub-grid scale turbulent stress.


WARNING
   none
****************************************/

#include <CCA/Components/Arches/SmagorinskyModel.h>


namespace Uintah {
class PhysicalConstants;
class BoundaryCondition;

class OdtClosure: public SmagorinskyModel {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for OdtClosure.
      OdtClosure(const ArchesLabel* label, 
		       const MPMArchesLabel* MAlb,
		       PhysicalConstants* phyConsts,
		       BoundaryCondition* bndryCondition);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for OdtClosure.
      virtual ~OdtClosure();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the initialization of the Smagorinsky Coefficient
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_initializeSmagCoeff(SchedulerP&,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels);
      
      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      void sched_initializeOdtvariable(SchedulerP&,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels);
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
			                 const TimeIntegratorLabel* timelabels,
                                               bool d_EKTCorrection);

      virtual void sched_computeScalarDissipation(SchedulerP&,
						  const PatchSet* patches,
						  const MaterialSet* matls,
			                 const TimeIntegratorLabel* timelabels,
                                                  bool d_EKTCorrection);

protected:
      int d_odtPoints; // number of odt Points at each LES cell
private:

      // GROUP: Constructors (not instantiated):
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for OdtClosure.
      OdtClosure();

      // GROUP: Action Methods (private)  :
      ///////////////////////////////////////////////////////////////////////

      void initializeSmagCoeff( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels);

      // Actually reCalculate the Turbulence sub model
      //    [in] 
      //        documentation here
      void initializeOdtvariable( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels);
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
			         const TimeIntegratorLabel* timelabels,
                                 bool d_EKTCorrection);

      void computeScalarDissipation(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw,
			            const TimeIntegratorLabel* timelabels,
                                    bool d_EKTCorrection);

private:
      double d_CF; //model constant
      double d_viscosity; // moleculor viscosity 
      // const VarLabel* variables 

}; // End class OdtClosure

/*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  
}

#endif

// $Log : $

