#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Components/Schedulers/MessageLog.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>
#include <vector>
#include <map>
using std::vector;

namespace Uintah {
   class Task;
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

   class MPIScheduler : public UintahParallelComponent, public Scheduler {
      struct SGArgs {
	 vector<int> dest;
	 vector<int> tags;
      };
      SGArgs sgargs; // THIS IS UGLY - Steve
      MessageLog log;
   public:
      MPIScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~MPIScheduler();
      
      virtual void problemSetup(const ProblemSpecP& prob_spec);
      
      //////////
      // Insert Documentation Here:
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void execute( const ProcessorGroup * pc,
			          DataWarehouseP   & old_dwp,
			          DataWarehouseP   & dwp );
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);

      //////////
      // Insert Documentation Here:
      virtual DataWarehouseP createDataWarehouse( DataWarehouseP& parent);
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      int numMatls);


       virtual LoadBalancer* getLoadBalancer();
       virtual void releaseLoadBalancer();
       
   private:
      void scatterParticles(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);
      void gatherParticles(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
      const VarLabel* reloc_old_posLabel;
      vector<vector<const VarLabel*> > reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      vector<vector<const VarLabel*> > reloc_new_labels;
      int reloc_numMatls;
      const VarLabel* scatterGatherVariable;

      MPIScheduler(const MPIScheduler&);
      MPIScheduler& operator=(const MPIScheduler&);

      TaskGraph graph;
      int d_generation;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/09/20 16:00:28  sparker
// Added external interface to LoadBalancer (for per-processor tasks)
// Added message logging functionality. Put the tag <MessageLog/> in
//    the ups file to enable
//
// Revision 1.6  2000/07/28 22:45:14  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.5  2000/07/28 03:01:54  rawat
// modified createDatawarehouse and added getTop()
//
// Revision 1.4  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.3  2000/07/26 20:14:11  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.2  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
//

#endif
