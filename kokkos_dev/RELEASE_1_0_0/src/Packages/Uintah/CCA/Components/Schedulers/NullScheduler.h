#ifndef UINTAH_HOMEBREW_NullSCHEDULER_H
#define UINTAH_HOMEBREW_NullSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <vector>
#include <map>
using std::vector;

namespace Uintah {

  class Task;
  class OnDemandDataWarehouse;

/**************************************

CLASS
   NullScheduler
   
   Short description...

GENERAL INFORMATION

   NullScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NullScheduler : public SchedulerCommon {
   public:
      NullScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~NullScheduler();
      
      //////////
      // Insert Documentation Here:
     virtual void execute( const ProcessorGroup * pc );
      
      //////////
      // Insert Documentation Here:
      virtual void advanceDataWarehouse(const GridP&);

      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      const MaterialSet* matls);

   private:
      NullScheduler(const NullScheduler&);
      NullScheduler& operator=(const NullScheduler&);

      const VarLabel* delt;
      bool firstTime;
   };
} // End namespace Uintah
   
#endif
