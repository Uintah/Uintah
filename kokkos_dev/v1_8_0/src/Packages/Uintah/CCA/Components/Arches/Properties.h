//----- Properties.h --------------------------------------------------

#ifndef Uintah_Component_Arches_Properties_h
#define Uintah_Component_Arches_Properties_h

/***************************************************************************
CLASS
    Properties
       Sets up the Properties ????
       
GENERAL INFORMATION
    Properties.h - Declaration of Properties class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
class MixingModel;
class Properties {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      // Constructor taking
      //   [in] 

      Properties(const ArchesLabel* label, const MPMArchesLabel* MAlb,
		 bool reactingFlow, bool enthalpySolver);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      // Destructor

      ~Properties();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database

      void problemSetup(const ProblemSpecP& params);

      // GROUP: Compute properties 
      ///////////////////////////////////////////////////////////////////////
      // Compute properties for inlet/outlet streams

      void computeInletProperties(const InletStream& inStream,
				  Stream& outStream);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of proprties

      void sched_computeProps(SchedulerP&, const PatchSet* patches,
			      const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of proprties for the first actual time 
      // step in an MPMArches run

      void sched_computePropsFirst_mm(SchedulerP&, const PatchSet* patches,
				      const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of proprties

      void sched_reComputeProps(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls);


      void sched_computePropsPred(SchedulerP&, const PatchSet* patches,
				  const MaterialSet* matls);

      void sched_computePropsInterm(SchedulerP&, const PatchSet* patches,
				  const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of density reference array here

      void sched_computeDenRefArray(SchedulerP&, const PatchSet* patches,
				    const MaterialSet* matls);

      void sched_computeDenRefArrayPred(SchedulerP&, const PatchSet* patches,
				    const MaterialSet* matls);

      void sched_computeDenRefArrayInterm(SchedulerP&, const PatchSet* patches,
				    const MaterialSet* matls);
      void sched_averageRKProps(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls,
				const int Runge_Kutta_current_step,
				const bool Runge_Kutta_last_step);
      void sched_reComputeRKProps(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls,
				const int Runge_Kutta_current_step,
				const bool Runge_Kutta_last_step);
      void sched_computeDrhodt(SchedulerP& sched, const PatchSet* patches,
				const MaterialSet* matls,
				const int Runge_Kutta_current_step,
				const bool Runge_Kutta_last_step);


      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the number of mixing variables

      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }

      ///////////////////////////////////////////////////////////////////////
      // Get the number of mixing variables

      inline int getNumMixStatVars() const{ 
	return d_numMixStatVars; 
      }

      // GROUP: Set Methods :
      ///////////////////////////////////////////////////////////////////////
      // Set the boundary consition pointer

      inline void setBC(BoundaryCondition* bc) {
	d_bc = bc;
      }

#ifdef PetscFilter
      inline void setFilter(Filter* filter) {
	d_filter = filter;
      }
#endif

protected :

private:

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      // Carry out actual computation of properties

      void computeProps(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Carry out actual recomputation of properties

      void reComputeProps(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

      void computePropsPred(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Carry out actual computation of properties for the first actual
      // time step in an MPMArches run

      void computePropsFirst_mm(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

      void computePropsInterm(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Carry out actual computation of density reference array

      void computeDenRefArray(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

      void computeDenRefArrayPred(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

      void computeDenRefArrayInterm(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

      void averageRKProps(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const int runge_kutta_current_step,
			  const bool runge_kutta_last_step);

      void reComputeRKProps(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const int runge_kutta_current_step,
			  const bool runge_kutta_last_step);

      void computeDrhodt(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const int runge_kutta_current_step,
			  const bool runge_kutta_last_step);

      // GROUP: Constructors Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const Properties&   

      Properties(const Properties&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const Properties&   

      Properties& operator=(const Properties&);

private:

      // Variable labels used by simulation controller
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      bool d_reactingFlow;
      bool d_enthalpySolve;
      bool d_radiationCalc;
      bool d_DORadiationCalc;
      bool d_flamelet;
      int d_numMixingVars;
      int d_numMixStatVars;
      double d_denUnderrelax;
      IntVector d_denRef;
      MixingModel* d_mixingModel;
      BoundaryCondition* d_bc;
#ifdef PetscFilter
      Filter* d_filter;
#endif
}; // end class Properties
} // End namespace Uintah


#endif

