#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
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
      
      // Makes and returns a map that maps strings to VarLabels of
      // that name and a list of material indices for which that
      // variable is valid (according to d_allcomps in graph).
      virtual VarLabelMaterialMap* makeVarLabelMaterialMap()
      { return graph.makeVarLabelMaterialMap(); }

      virtual const vector<const Task::Dependency*>& getInitialRequires()
      { return graph.getInitialRequires(); }
     
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

      double d_lasttime;
      vector<char*> d_labels;
      vector<double> d_times;
      void emitTime(char* label);
      void emitTime(char* label, double time);
   };
} // End namespace Uintah
   
#endif
