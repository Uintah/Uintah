//----- CO2RateSrc.h -----------------------------------------------

#ifndef Uintah_Component_Arches_CO2RateSrc_h
#define Uintah_Component_Arches_CO2RateSrc_h

#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSrc.h>

/**************************************
CLASS
   CO2RateSrc
   
   Class CO2RateSrc is 

GENERAL INFORMATION
   CO2RateSrc.h - declaration of the class
   
   Author: Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   July 30th , 2007
   
   C-SAFE 
   
   Copyright U of U 2007

KEYWORDS


DESCRIPTION
   Class CO2RateSrc is 

WARNING
   none
****************************************/

namespace Uintah {
class CO2RateSrc: public ExtraScalarSrc{

public:

      ////////////////////////////////////////////////////////////////////////
      // Constructor for CO2RateSrc.
      CO2RateSrc(const ArchesLabel* label, 
		     const MPMArchesLabel* MAlb,
                     const VarLabel* d_src_label);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for CO2RateSrc.
      ~CO2RateSrc();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up of the problem specification database
      void problemSetup(const ProblemSpecP& db);


      void sched_addExtraScalarSrc(SchedulerP& sched, 
                                   const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels);
protected:

private:
      CO2RateSrc();


      void addExtraScalarSrc(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels);

}; // End class CO2RateSrc
} // End namespace Uintah

#endif


