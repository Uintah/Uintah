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

#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

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
			                 const TimeIntegratorLabel* timelabels);

      virtual void sched_computeScalarDissipation(SchedulerP&,
						  const PatchSet* patches,
						  const MaterialSet* matls,
			                 const TimeIntegratorLabel* timelabels);

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
			         const TimeIntegratorLabel* timelabels);

      void computeScalarDissipation(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw,
			            const TimeIntegratorLabel* timelabels);

private:
      double d_CF; //model constant
      double d_viscosity; // moleculor viscosity 
      // const VarLabel* variables 

}; // End class OdtClosure

/*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  struct odtData { 
	  	double x_x[10]; //x coordinates along x direction line
		double y_y[10]; //y coordinates along x direction line
		double z_z[10]; //z coordinates along x direction line
		double x_u[10]; //u velocity along x direction line
  		double x_v[10]; //v velocity along x direction line
		double x_w[10]; //w velocity along x direction line
		double x_rho[10]; //density along x direction line 
		double x_T[10]; //temperature along x direction line
		double x_Phi[10]; //scalar  along x direction line
		double y_u[10]; //u velocity along y direction line
  		double y_v[10]; //v velocity along y direction line
		double y_w[10]; //w velocity along y direction line
		double y_rho[10]; //density along y direction line 
		double y_T[10]; //temperature along y direction line
		double y_Phi[10]; //scalar  along y direction line
		double z_u[10]; //u velocity along z direction line
  		double z_v[10]; //v velocity along z direction line
		double z_w[10]; //w velocity along z direction line
		double z_rho[10]; //density along z direction line 
		double z_T[10]; //temperature along z direction line
		double z_Phi[10]; //scalar  along z direction line

  };    


//________________________________________________________________________
  const TypeDescription* fun_getTypeDescription(odtData*);    

} // End namespace Uintah
  
namespace SCIRun {
  void swapbytes( Uintah::odtData& d);
}       
  

#endif

// $Log : $

