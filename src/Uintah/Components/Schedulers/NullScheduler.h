#ifndef UINTAH_HOMEBREW_NullSCHEDULER_H
#define UINTAH_HOMEBREW_NullSCHEDULER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
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

   class NullScheduler : public UintahParallelComponent, public Scheduler {
   public:
      NullScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~NullScheduler();
      
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
      virtual DataWarehouseP createDataWarehouse(DataWarehouseP& parent_dw);
      
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

   private:
      NullScheduler(const NullScheduler&);
      NullScheduler& operator=(const NullScheduler&);

      TaskGraph graph;
      // id of datawarehouse
      int d_generation;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/12/10 09:06:10  sparker
// Merge from csafe_risky1
//
// Revision 1.1.2.1  2000/10/10 05:33:12  sparker
// Added NullScheduler (used for optimizing taskgraph construction)
//
//

#endif
