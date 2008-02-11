//----- ExtraScalarSrc.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExtraScalarSrc_h
#define Uintah_Component_Arches_ExtraScalarSrc_h

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

/**************************************
CLASS
   ExtraScalarSrc
   
   Class ExtraScalarSrc is an abstract base class

GENERAL INFORMATION
   ExtraScalarSrc.h - declaration of the class
   
   Author: Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   July 30th , 2007
   
   C-SAFE 
   
   Copyright U of U 2007

KEYWORDS


DESCRIPTION
   Class ExtraScalarSrc is an abstract type

WARNING
   none
****************************************/

namespace Uintah {
class TimeIntegratorLabel;
class ExtraScalarSrc {

public:

      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for ExtraScalarSrc.
      ExtraScalarSrc(const ArchesLabel* label, 
		     const MPMArchesLabel* MAlb,
                     const VarLabel* d_src_label);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for ExtraScalarSrc.
      virtual ~ExtraScalarSrc();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up of the problem specification database
      virtual void problemSetup(const ProblemSpecP& db) = 0;


      virtual void sched_addExtraScalarSrc(SchedulerP& sched, 
                                           const PatchSet* patches,
					   const MaterialSet* matls,
				     const TimeIntegratorLabel* timelabels) = 0;

      virtual inline const string getTableName() = 0;
      virtual inline void setTableIndex(int tableIndex) = 0;  
					     
protected:
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      const VarLabel* d_scalar_nonlin_src_label;

private:

}; // End class ExtraScalarSrc
} // End namespace Uintah

#endif


