#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <vector>
#include <map>
using std::vector;

namespace Uintah {
  struct SendRecord;
  class SendState;
  class DetailedTasks;
  class Task;
  class OnDemandDataWarehouse;

/**************************************

CLASS
   MPIScheduler
   
   Short description...

GENERAL INFORMATION

   MPIScheduler.h

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

   class MPIScheduler : public SchedulerCommon {
      MessageLog log;
   public:
      MPIScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~MPIScheduler();
      
      virtual void problemSetup(const ProblemSpecP& prob_spec);
      
      //////////
      // Insert Documentation Here:
     virtual void compile( const ProcessorGroup * pc );
     virtual void execute( const ProcessorGroup * pc);
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      const MaterialSet* matls);


   private:
      void relocateParticles(const ProcessorGroup*,
			     const PatchSubset* patch,
			     const MaterialSubset* patch,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw);
      const VarLabel* reloc_old_posLabel;
      vector<vector<const VarLabel*> > reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      vector<vector<const VarLabel*> > reloc_new_labels;
      const MaterialSet* reloc_matls;

      MPIScheduler(const MPIScheduler&);
      MPIScheduler& operator=(const MPIScheduler&);

      double d_lasttime;
      vector<char*> d_labels;
      vector<double> d_times;
      void emitTime(char* label);
      void emitTime(char* label, double time);
   };
} // End namespace Uintah
   
#endif
