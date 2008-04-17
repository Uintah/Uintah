//----- SO2RateSrc.h -----------------------------------------------

#ifndef Uintah_Component_Arches_SO2RateSrc_h
#define Uintah_Component_Arches_SO2RateSrc_h

#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSrc.h>

/**************************************
CLASS
   SO2RateSrc
   
   Class SO2RateSrc is 

GENERAL INFORMATION
   SO2RateSrc.h - declaration of the class
   
   Author: Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   July 30th , 2007
   
   C-SAFE 
   
   Copyright U of U 2007

KEYWORDS


DESCRIPTION
   Class SO2RateSrc is 

WARNING
   none
****************************************/

namespace Uintah {
class BoundaryCondition; 
class SO2RateSrc: public ExtraScalarSrc{

public:

      ////////////////////////////////////////////////////////////////////////
      // Constructor for SO2RateSrc.
      SO2RateSrc(const ArchesLabel* label, 
		     const MPMArchesLabel* MAlb,
                     const VarLabel* d_src_label);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for SO2RateSrc.
      ~SO2RateSrc();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up of the problem specification database
      void problemSetup(const ProblemSpecP& db);


      void sched_addExtraScalarSrc(SchedulerP& sched, 
                                   const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels);

      inline const string getTableName(){
	      return d_tableName;
      }
/*
      inline const int getTableIndex(){
	      return d_tableIndex;
      }
*/
      inline void setTableIndex(int tableIndex){
	      d_tableIndex = tableIndex;
      }


protected:

private:
      SO2RateSrc();


      void addExtraScalarSrc(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels);

	string	d_tableName;
	int d_tableIndex;

}; // End class SO2RateSrc
} // End namespace Uintah

#endif


