
#ifndef UINTAH_HOMEBREW_SCHEDULER_H
#define UINTAH_HOMEBREW_SCHEDULER_H

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

#include <string>
#include <vector>
#include <list>
#include <map>

class DOM_Document;
class DOM_Element;

namespace Uintah {

class VarLabel;
class ProcessorGroup;

using std::vector;
using std::string;
using std::list;
using std::map;

class LoadBalancer;
class Task;
class TaskGraph;
class VarLabel;
class ProcessorGroup;

/**************************************

CLASS
   Scheduler
   
   Short description...

GENERAL INFORMATION

   Scheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Scheduler : public UintahParallelPort {
  public:
    Scheduler(Output* oport);
    virtual ~Scheduler();

    virtual void problemSetup(const ProblemSpecP& prob_spec);


    void doEmitTaskGraphDocs()
    { m_doEmitTaskGraphDocs = true; }
    
    //////////
    // Insert Documentation Here:
    virtual void initialize() = 0;
       
    //////////
    // Insert Documentation Here:
    virtual void execute(const ProcessorGroup * pc, 
			 DataWarehouseP   & old_dwp,
			 DataWarehouseP   & dwp ) = 0;
       
    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t) = 0;

    virtual const vector<const Task::Dependency*>& getInitialRequires() = 0;

    virtual LoadBalancer* getLoadBalancer() = 0;
    virtual void releaseLoadBalancer() = 0;
       
    //////////
    // Insert Documentation Here:
    virtual DataWarehouseP createDataWarehouse(DataWarehouseP& parent_dw) = 0;
    //    protected:

    //////////
    // Insert Documentation Here:
    virtual void
    scheduleParticleRelocation(const LevelP& level,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw,
			       const VarLabel* posLabel,
			       const vector<vector<const VarLabel*> >& labels,
			       const VarLabel* new_posLabel,
			       const vector<vector<const VarLabel*> >& new_labels,
			       int numMatls) = 0;

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    typedef map< string, list<int> > VarLabelMaterialMap;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap() = 0;
  protected:
    void makeTaskGraphDoc(const vector<Task*>& tasks,
			  bool emit_edges = true);
    void emitNode(const Task* name, time_t start, double duration);
    void finalizeNodes(int process=0);
  
  private:
    Scheduler(const Scheduler&);
    Scheduler& operator=(const Scheduler&);

    Output* m_outPort;
    DOM_Document* m_graphDoc;
    DOM_Element* m_nodes;

    bool m_doEmitTaskGraphDocs;
    //unsigned int m_executeCount;
  };

} // End namespace Uintah

#endif
