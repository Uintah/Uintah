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

namespace Uintah {
class TurbulenceModel
{
public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for TurbulenceModel.
      TurbulenceModel();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for TurbulenceModel.
      virtual ~TurbulenceModel();

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

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Schedule the computation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_computeTurbSubmodel(SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls) = 0;

      ///////////////////////////////////////////////////////////////////////
      // Interface for Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls) = 0;

      virtual void sched_computeScalarVariance(SchedulerP&, const PatchSet* patches,
					       const MaterialSet* matls) = 0;
      // GROUP: Action Computations :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Calculate the wall velocity boundary conditions
      //    [in] 
      //        index = documentation here
      virtual void calcVelocityWallBC(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index,
				      int eqnType) = 0;

      ///////////////////////////////////////////////////////////////////////
      // Interface for Calculate the velocity source terms
      //    [in] 
      //        index = documentation here
      virtual void calcVelocitySource(const ProcessorGroup*,
				      const Patch* patch,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw, 
				      int index) = 0;

private:

}; // End class TurbulenceModel
} // End namespace Uintah
  
  

#endif

// $Log :$


