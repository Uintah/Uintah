#ifndef UINTAH_HOMEBREW_DetailedTasks_H
#define UINTAH_HOMEBREW_DetailedTasks_H

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabelMatlPatchDW.h>
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
  class DetailedTask;
  class DetailedTasks;
  class TaskGraph;
  class SchedulerCommon;
  
  class DetailedDep {
  public:
    DetailedDep(DetailedDep* next, Task::Dependency* comp,
		Task::Dependency* req, DetailedTask* toTask,
		const Patch* fromPatch, int matl,
		const IntVector& low, const IntVector& high)
      : next(next), comp(comp), req(req),                  
        fromPatch(fromPatch), low(low), high(high), matl(matl)
    {
      ASSERT(Min(high - low, IntVector(1, 1, 1)) == IntVector(1, 1, 1));
      Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(), true);
      ASSERT(fromPatch == 0 || (Min(low, fromPatch->getLowIndex(basis, req->var->getBoundaryLayer())) ==
				fromPatch->getLowIndex(basis, req->var->getBoundaryLayer())));
      ASSERT(fromPatch == 0 || (Max(high, fromPatch->getHighIndex(basis, req->var->getBoundaryLayer())) ==
				fromPatch->getHighIndex(basis, req->var->getBoundaryLayer())));
      toTasks.push_back(toTask);
    }

    // As an arbitrary convention, non-data dependency have a NULL fromPatch.
    // These types of dependency exist between a modifying task and any task
    // that requires the data (from ghost cells in particular) before it is
    // modified preventing the possibility of modifying data while it is being
    // used.
    bool isNonDataDependency() const
    { return (fromPatch == NULL); }
    
    DetailedDep* next;
    Task::Dependency* comp;
    Task::Dependency* req;
    list<DetailedTask*> toTasks;
    const Patch* fromPatch;
    IntVector low, high;
    int matl;
  };

  class DependencyBatch {
  public:
    DependencyBatch(int to, DetailedTask* fromTask, DetailedTask* toTask)
      : comp_next(0), fromTask(fromTask),
	head(0), messageTag(-1), to(to), 
	received_(false), madeMPIRequest_(false),
	lock_(0)
    {
      toTasks.push_back(toTask);
    }
    ~DependencyBatch();

    // The first thread calling this will return true, all others
    // will return false.
    bool makeMPIRequest();

    // Tells this batch that it has actually been received and
    // awakens anybody blocked in makeMPIRequest().
    void received(const ProcessorGroup * pg);
    bool wasReceived()
    { return received_; }

    // Initialize receiving information for makeMPIRequest() and received()
    // so that it can receive again.
    void reset();

    void addReceiveListener(int mpiSignal);
    
    //DependencyBatch* req_next;
    DependencyBatch* comp_next;
    DetailedTask* fromTask;
    list<DetailedTask*> toTasks;
    DetailedDep* head;
    int messageTag;
    int to;

  private:
    volatile bool received_;
    volatile bool madeMPIRequest_;
    Mutex* lock_;
    set<int> receiveListeners_;

    DependencyBatch(const DependencyBatch&);
    DependencyBatch& operator=(const DependencyBatch&);
  };

  struct InternalDependency {
    InternalDependency(DetailedTask* prerequisiteTask,
		       DetailedTask* dependentTask, const VarLabel* var,
		       long satisfiedGeneration)
      : prerequisiteTask(prerequisiteTask), dependentTask(dependentTask),
	satisfiedGeneration(satisfiedGeneration)
    { addVarLabel(var); }

    void addVarLabel(const VarLabel* var)
    { vars.insert(var); }
    
    DetailedTask* prerequisiteTask;
    DetailedTask* dependentTask;
    set<const VarLabel*, VarLabel::Compare> vars;
    unsigned long satisfiedGeneration;
  };

  struct ScrubItem {
    ScrubItem* next;
    const VarLabel* var;
    int dw;
    ScrubItem(ScrubItem* next, const VarLabel* var, int dw)
      : next(next), var(var), dw(dw)
    {}
  };

  typedef map<VarLabelMatlPatchDW, int> ScrubCountMap;
  
  class DetailedTask {
  public:
    DetailedTask(Task* task, const PatchSubset* patches,
		 const MaterialSubset* matls, DetailedTasks* taskGroup);
    ~DetailedTask();
    
    void doit(const ProcessorGroup* pg, vector<OnDemandDataWarehouseP>& oddws,
	      vector<DataWarehouseP>& dws);
    // Called after doit and mpi data sent (packed in buffers) finishes.
    // Handles internal dependencies and scrubbing.
    // Called after doit finishes.
    void done(vector<OnDemandDataWarehouseP>& dws);

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

    map<DependencyBatch*, DependencyBatch*>& getRequires() {
      return reqs;
    }
    DependencyBatch* getComputes() const {
      return comp_head;
    }

    void findRequiringTasks(const VarLabel* var,
			    list<DetailedTask*>& requiringTasks);

    void emitEdges(ProblemSpecP edgesElement);

    bool addRequires(DependencyBatch*);
    void addComputes(DependencyBatch*);

    void addInternalDependency(DetailedTask* prerequisiteTask,
			       const VarLabel* var);

    bool areInternalDependenciesSatisfied()
    { return (numPendingInternalDependencies == 0); }
  protected:
    friend class TaskGraph;
  private:
    // called by done()
    void scrub(vector<OnDemandDataWarehouseP>&);

    Task* task;
    const PatchSubset* patches;
    const MaterialSubset* matls;
    map<DependencyBatch*, DependencyBatch*> reqs;
    DependencyBatch* comp_head;
    DetailedTasks* taskGroup;

    mutable string name_; /* doesn't get set until getName() is called
			     the first time. */

    // Called when prerequisite tasks (dependencies) call done.
    void dependencySatisfied(InternalDependency* dep);

    
    // Internal dependencies are dependencies within the same process.
    list<InternalDependency> internalDependencies;
    
    // internalDependents will point to InternalDependency's in the
    // internalDependencies list of the requiring DetailedTasks.
    map<DetailedTask*, InternalDependency*> internalDependents;
    
    unsigned long numPendingInternalDependencies;
    Mutex internalDependencyLock;
    
    int resourceIndex;

    DetailedTask(const Task&);
    DetailedTask& operator=(const Task&);
  };

  class DetailedTasks {
  public:
    DetailedTasks(SchedulerCommon* sc, const ProcessorGroup* pg,
		  const TaskGraph* taskgraph,
		  bool mustConsiderInternalDependencies = false);
    ~DetailedTasks();

    void add(DetailedTask* task);
    int numTasks() const {
      return (int)tasks_.size();
    }
    DetailedTask* getTask(int i) {
      return tasks_[i];
    }


    void assignMessageTags(int me);

    void initializeScrubs(vector<OnDemandDataWarehouseP>& dws);

    void possiblyCreateDependency(DetailedTask* from, Task::Dependency* comp,
				  const Patch* fromPatch,
				  DetailedTask* to, Task::Dependency* req,
				  const Patch* toPatch, int matl,
				  const IntVector& low, const IntVector& high);

    DetailedTask* getOldDWSendTask(int proc);

    void logMemoryUse(ostream& out, unsigned long& total,
		      const std::string& tag);

    void initTimestep();
    
    void computeLocalTasks(int me);
    int numLocalTasks() const {
      return (int)localtasks_.size();
    }

    DetailedTask* localTask(int idx) {
      return localtasks_[idx];
    }

    void emitEdges(ProblemSpecP edgesElement, int rank);

    DetailedTask* getNextInternalReadyTask();
    
    void createScrubCounts();

    bool mustConsiderInternalDependencies()
    { return mustConsiderInternalDependencies_; }

    unsigned long getCurrentDependencyGeneration()
    { return currentDependencyGeneration_; }

    const TaskGraph* getTaskGraph() const {
      return taskgraph_;
    }
    void setScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw,
		       vector<OnDemandDataWarehouseP>& dws);
  protected:
    friend class DetailedTask;

    void internalDependenciesSatisfied(DetailedTask* task);
    SchedulerCommon* getSchedulerCommon() {
      return sc_;
    }
  private:
    void initializeBatches();

    void incrementDependencyGeneration();

    void addScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw);
    bool getScrubCount(const VarLabel* var, int matlindex,
		       const Patch* patch, int dw, int& count);

    SchedulerCommon* sc_;
    const ProcessorGroup* d_myworld;
    vector<DetailedTask*> tasks_;
#if 0
    vector<DetailedReq*> initreqs_;
#endif
    const TaskGraph* taskgraph_;
    vector<Task*> stasks_;
    vector<DetailedTask*> localtasks_;
    vector<DependencyBatch*> batches_;
    DetailedDep* initreq_;
    
    // True for mixed scheduler which needs to keep track of internal
    // depedencies.
    bool mustConsiderInternalDependencies_;

    // In the future, we may want to prioritize tasks for the MixedScheduler
    // to run.  I implemented this using topological sort order as the priority
    // but that probably isn't a good way to do unless you make it a breadth
    // first topological order.
    //typedef priority_queue<DetailedTask*, vector<DetailedTask*>, TaskNumberCompare> TaskQueue;    
    typedef queue<DetailedTask*> TaskQueue;
    
    TaskQueue readyTasks_; 
    TaskQueue initiallyReadyTasks_;

    // This "generation" number is to keep track of which InternalDependency
    // links have been satisfied in the current timestep and avoids the
    // need to traverse all InternalDependency links to reset values.
    unsigned long currentDependencyGeneration_;
    Mutex readyQueueMutex_;
    Semaphore readyQueueSemaphore_;

    ScrubCountMap scrubCountMap_;

    DetailedTasks(const DetailedTasks&);
    DetailedTasks& operator=(const DetailedTasks&);
  };
} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::DetailedTask& task);
std::ostream& operator<<(std::ostream& out, const Uintah::DetailedDep& task);

#endif

