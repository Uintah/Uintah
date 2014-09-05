
#ifndef UINTAH_HOMEBREW_SCHEDULERCOMMON_H
#define UINTAH_HOMEBREW_SCHEDULERCOMMON_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>

class DOM_Document;
class DOM_Element;

namespace Uintah {
  using namespace std;
  class Output;
  class DetailedTask;
  class DetailedTasks;
  class OnDemandDataWarehouse;

/**************************************

CLASS
   SchedulerCommon
   
   Short description...

GENERAL INFORMATION

   SchedulerCommon.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SchedulerCommon

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SchedulerCommon : public UintahParallelComponent, public Scheduler {
  public:
    SchedulerCommon(const ProcessorGroup* myworld, Output* oport);
    virtual ~SchedulerCommon();

    virtual void problemSetup(const ProblemSpecP& prob_spec);

    virtual void doEmitTaskGraphDocs();

    //////////
    // Insert Documentation Here:
    virtual void initialize();
       
    virtual void compile( const ProcessorGroup * pc );

    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t, const PatchSet*, const MaterialSet*);

    virtual const vector<const Task::Dependency*>& getInitialRequires();

    virtual LoadBalancer* getLoadBalancer();
    virtual void releaseLoadBalancer();
       
    virtual DataWarehouse* get_old_dw();
    virtual DataWarehouse* get_new_dw();
      
    //////////
    // Insert Documentation Here:
    virtual void advanceDataWarehouse(const GridP& grid);

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    typedef map< string, list<int> > VarLabelMaterialMap;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap();
  protected:
    void makeTaskGraphDoc(const DetailedTasks* dt,
			  bool emit_edges = true);
    void emitNode(const DetailedTask* dt, double start, double duration);
    void finalizeNodes(int process=0);
    
    TaskGraph graph;
    int d_generation;
    OnDemandDataWarehouse* dw[2];
    DetailedTasks* dt;
  private:
    SchedulerCommon(const SchedulerCommon&);
    SchedulerCommon& operator=(const SchedulerCommon&);

    Output* m_outPort;
    DOM_Document* m_graphDoc;
    DOM_Element* m_nodes;
    bool emit_taskgraph;
  };
} // End namespace Uintah

#endif
