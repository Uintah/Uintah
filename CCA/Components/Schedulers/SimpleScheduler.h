#ifndef UINTAH_HOMEBREW_SimpleSCHEDULER_H
#define UINTAH_HOMEBREW_SimpleSCHEDULER_H

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
   SimpleScheduler
   
   Short description...

GENERAL INFORMATION

   SimpleScheduler.h

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

   class SimpleScheduler : public SchedulerCommon {
   public:
      SimpleScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~SimpleScheduler();
      
      //////////
      // Insert Documentation Here:
     virtual void compile( const ProcessorGroup * pc );
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

      void relocateParticles(const ProcessorGroup*,
			     const PatchSubset* patch,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw);
      const VarLabel* scatterGatherVariable;
      SimpleScheduler(const SimpleScheduler&);
      SimpleScheduler& operator=(const SimpleScheduler&);

     vector<Task*> tasks;
   };

} // End namespace Uintah
   
#endif
