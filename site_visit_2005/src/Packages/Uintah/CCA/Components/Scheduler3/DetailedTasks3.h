#ifndef UINTAH_HOMEBREW_DetailedTasks3_H
#define UINTAH_HOMEBREW_DetailedTasks3_H

#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3P.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
//#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabelMatlDW.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ConditionVariable.h>
#include <sgi_stl_warnings_off.h>
#include <list>
#include <queue>
#include <vector>
#include <map>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using SCIRun::Min;
  using SCIRun::Max;
  using SCIRun::Mutex;
  using SCIRun::Semaphore;

  class ProcessorGroup;
  class DataWarehouse;
  class DetailedTask3;
  class DetailedTasks3;
  class TaskGraph3;
  class Scheduler3Common;
  
  typedef map<VarLabelMatlDW<Patch>, int> ScrubCountMap;
  
  class DetailedTask3 {
  public:
    DetailedTask3(Task* task, const PatchSubset* patches,
		 const MaterialSubset* matls, DetailedTasks3* taskGroup);
    ~DetailedTask3();
    
    void doit(const ProcessorGroup* pg, vector<PatchBasedDataWarehouse3P>& oddws,
	      vector<DataWarehouseP>& dws);
    // Called after doit and mpi data sent (packed in buffers) finishes.
    // Handles internal dependencies and scrubbing.
    // Called after doit finishes.
    void done(vector<PatchBasedDataWarehouse3P>& dws);

    string getName() const;
    
    const Task* getTask() const {
      return task;
    }
    const PatchSubset* getPatches() const {
      return patches;
    }
    const MaterialSubset* getMaterials() const {
      return matls;
    }
    void assignResource( int idx ) {
      resourceIndex = idx;
    }
    int getAssignedResourceIndex() const {
      return resourceIndex;
    }

    void findRequiringTasks(const VarLabel* var,
			    list<DetailedTask3*>& requiringTasks);


    void emitEdges(ProblemSpecP edgesElement);

    /*    map<DependencyBatch*, DependencyBatch*>& getRequires() {
      return reqs;
    }
    DependencyBatch* getComputes() const {
      return comp_head;
    }

    bool addRequires(DependencyBatch*);
    void addComputes(DependencyBatch*);
    */

    void addInternalDependency(DetailedTask3* prerequisiteTask,
			       const VarLabel* var);

    bool areInternalDependenciesSatisfied() { return false;}
  protected:
    friend class TaskGraph3;
  private:
    // called by done()
    void scrub(vector<PatchBasedDataWarehouse3P>&);

    Task* task;
    const PatchSubset* patches;
    const MaterialSubset* matls;
    //map<DependencyBatch*, DependencyBatch*> reqs;
    //DependencyBatch* comp_head;
    DetailedTasks3* taskGroup;

    mutable string name_; /* doesn't get set until getName() is called
			     the first time. */

    int resourceIndex;

    DetailedTask3(const Task&);
    DetailedTask3& operator=(const Task&);
  };

  class DetailedTasks3 {
  public:
    DetailedTasks3(Scheduler3Common* sc, const ProcessorGroup* pg,
		  const TaskGraph3* taskgraph,
		  bool mustConsiderInternalDependencies = false);
    ~DetailedTasks3();

    void add(DetailedTask3* task);
    int numTasks() const {
      return (int)tasks_.size();
    }
    DetailedTask3* getTask(int i) {
      return tasks_[i];
    }


    void assignMessageTags(int me);

    void initializeScrubs(vector<PatchBasedDataWarehouse3P>& dws);

    void possiblyCreateDependency(DetailedTask3* from, Task::Dependency* comp,
				  const Patch* fromPatch,
				  DetailedTask3* to, Task::Dependency* req,
				  const Patch* toPatch, int matl,
				  const IntVector& low, const IntVector& high);

    DetailedTask3* getOldDWSendTask(int proc);

    void logMemoryUse(ostream& out, unsigned long& total,
		      const std::string& tag);

    void initTimestep();
    
    void computeLocalTasks(int me);
    int numLocalTasks() const {
      return (int)localtasks_.size();
    }

    DetailedTask3* localTask(int idx) {
      return localtasks_[idx];
    }

    void emitEdges(ProblemSpecP edgesElement, int rank);

    DetailedTask3* getNextInternalReadyTask();
    
    void createScrubCounts();

    bool mustConsiderInternalDependencies()
    { return mustConsiderInternalDependencies_; }

    unsigned long getCurrentDependencyGeneration()
    { return currentDependencyGeneration_; }

    const TaskGraph3* getTaskGraph() const {
      return taskgraph_;
    }
    void setScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw,
		       vector<PatchBasedDataWarehouse3P>& dws);
  protected:
    friend class DetailedTask3;

    void internalDependenciesSatisfied(DetailedTask3* task);
    Scheduler3Common* getSchedulerCommon() {
      return sc_;
    }
  private:
    void initializeBatches();

    void incrementDependencyGeneration();

    void addScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw);
    bool getScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw, int& count);

    Scheduler3Common* sc_;
    const ProcessorGroup* d_myworld;
    vector<DetailedTask3*> tasks_;
#if 0
    vector<DetailedReq*> initreqs_;
#endif
    const TaskGraph3* taskgraph_;
    vector<Task*> stasks_;
    vector<DetailedTask3*> localtasks_;
    //vector<DependencyBatch*> batches_;
    //DetailedDep* initreq_;
    
    // True for mixed scheduler which needs to keep track of internal
    // depedencies.
    bool mustConsiderInternalDependencies_;

    // In the future, we may want to prioritize tasks for the MixedScheduler
    // to run.  I implemented this using topological sort order as the priority
    // but that probably isn't a good way to do unless you make it a breadth
    // first topological order.
    //typedef priority_queue<DetailedTask3*, vector<DetailedTask3*>, TaskNumberCompare> TaskQueue;    
    typedef queue<DetailedTask3*> TaskQueue;
    
    TaskQueue readyTasks_; 
    TaskQueue initiallyReadyTasks_;

    // This "generation" number is to keep track of which InternalDependency
    // links have been satisfied in the current timestep and avoids the
    // need to traverse all InternalDependency links to reset values.
    unsigned long currentDependencyGeneration_;
    Mutex readyQueueMutex_;
    Semaphore readyQueueSemaphore_;

    ScrubCountMap scrubCountMap_;

    DetailedTasks3(const DetailedTasks3&);
    DetailedTasks3& operator=(const DetailedTasks3&);
  };
} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::DetailedTask3& task);

#endif

