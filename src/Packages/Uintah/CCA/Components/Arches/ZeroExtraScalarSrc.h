//----- ZeroExtraScalarSrc.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ZeroExtraScalarSrc_h
#define Uintah_Component_Arches_ZeroExtraScalarSrc_h

#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSrc.h>

/**************************************
CLASS
   ZeroExtraScalarSrc
   
   Class ZeroExtraScalarSrc is 

GENERAL INFORMATION
   ZeroExtraScalarSrc.h - declaration of the class
   
   Author: Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   July 30th , 2007
   
   C-SAFE 
   
   Copyright U of U 2007

KEYWORDS


DESCRIPTION
   Class ZeroExtraScalarSrc is 

WARNING
   none
****************************************/

namespace Uintah {
class ZeroExtraScalarSrc: public ExtraScalarSrc{

public:

  ////////////////////////////////////////////////////////////////////////
  // Constructor for ZeroExtraScalarSrc.
  ZeroExtraScalarSrc(const ArchesLabel* label, 
                     const MPMArchesLabel* MAlb,
                     const VarLabel* d_src_label);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for ZeroExtraScalarSrc.
  ~ZeroExtraScalarSrc();


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

  inline void setTableIndex(int tableIndex){
    d_tableIndex = tableIndex;
  }

protected:

private:
  ZeroExtraScalarSrc();


  void addExtraScalarSrc(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const TimeIntegratorLabel* timelabels);

  string d_tableName;
  int d_tableIndex; 

}; // End class ZeroExtraScalarSrc
} // End namespace Uintah

#endif


