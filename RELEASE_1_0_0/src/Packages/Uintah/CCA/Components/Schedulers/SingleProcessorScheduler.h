#ifndef UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H
#define UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <vector>
#include <map>
using std::vector;

namespace Uintah {

  class OnDemandDataWarehouse;
  class Task;

/**************************************

CLASS
   SingleProcessorScheduler
   
   Short description...

GENERAL INFORMATION

   SingleProcessorScheduler.h

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

   class SingleProcessorScheduler : public SchedulerCommon {
   public:
      SingleProcessorScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~SingleProcessorScheduler();
      
      //////////
      // Insert Documentation Here:
     virtual void execute( const ProcessorGroup * pc );
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      const MaterialSet* matls);

   private:
      const VarLabel* reloc_old_posLabel;
      vector<vector<const VarLabel*> > reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      vector<vector<const VarLabel*> > reloc_new_labels;
      const MaterialSet* reloc_matls;

      void scatterParticles(const ProcessorGroup*,
			    const PatchSubset* patch,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);
      void gatherParticles(const ProcessorGroup*,
			   const PatchSubset* patch,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);
      const VarLabel* scatterGatherVariable;
      SingleProcessorScheduler(const SingleProcessorScheduler&);
      SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);
   };

} // End namespace Uintah
   
#endif
