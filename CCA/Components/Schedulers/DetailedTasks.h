#ifndef UINTAH_HOMEBREW_DetailedTasks_H
#define UINTAH_HOMEBREW_DetailedTasks_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ConditionVariable.h>
#include <list>
#include <vector>

namespace Uintah {
  class ProcessorGroup;
  class DataWarehouse;
  class DetailedTask;
  class DetailedTasks;

  class DetailedDep {
  public:
    DetailedDep(DetailedDep* next, Task::Dependency* comp,
		Task::Dependency* req, DetailedTask* toTask,
		const Patch* fromPatch, int matl,
		const IntVector& low, const IntVector& high)
      : next(next), comp(comp), req(req),                  
        fromPatch(fromPatch), low(low), high(high), matl(matl)
    {
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
	lock_(0), cv_(0)
    {
      toTasks.push_back(toTask);
    }
    ~DependencyBatch();

    // The first thread calling this will return true, all others
    // will return false.
    bool makeMPIRequest();

    // The first thread calling this will return true, all others
    // will block until received() is called and return false.
    bool waitForMPIRequest();

    // Tells this batch that it has actually been received and
    // awakens anybody blocked in makeMPIRequest().
    void received();

    // Initialize receiving information for makeMPIRequest() and received()
    // so that it can receive again.
    void reset();

    //DependencyBatch* req_next;
    DependencyBatch* comp_next;
    DetailedTask* fromTask;
    list<DetailedTask*> toTasks;
    DetailedDep* head;
    int messageTag;
    int to;

  private:
    bool received_;
    bool madeMPIRequest_;
    Mutex* lock_;
    ConditionVariable* cv_;

    DependencyBatch(const DependencyBatch&);
    DependencyBatch& operator=(const DependencyBatch&);
  };

  struct InternalDependency {
    InternalDependency(DetailedTask* prerequisiteTask, DetailedTask* dependentTask,
		       long satisfiedGeneration)
      : prerequisiteTask(prerequisiteTask), dependentTask(dependentTask),
	satisfiedGeneration(satisfiedGeneration)
    { }
    
    DetailedTask* prerequisiteTask;
    DetailedTask* dependentTask;
    unsigned long satisfiedGeneration;
  };

  struct ScrubItem {
    ScrubItem* next;
    const VarLabel* var;
    Task::WhichDW dw;
    ScrubItem(ScrubItem* next, const VarLabel* var, Task::WhichDW dw)
      : next(next), var(var), dw(dw)
    {}
  };

  class DetailedTask {
  public:
    DetailedTask(Task* task, const PatchSubset* patches,
		 const MaterialSubset* matls, DetailedTasks* taskGroup);
    ~DetailedTask();
    
    void doit(const ProcessorGroup* pg, DataWarehouse* old_dw,
	      DataWarehouse* new_dw);
    // Called after doit finishes.
    void done();

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
    const ScrubItem* getScrublist() const {
      return scrublist;
    }

    bool addRequires(DependencyBatch*);
    void addComputes(DependencyBatch*);
    void addScrub(const VarLabel* var, Task::WhichDW dw);

    void addInternalDependency(DetailedTask* prerequisiteTask);

    bool areInternalDependenciesSatisfied()
    { return (numPendingInternalDependencies == 0); }
  protected:
    friend class TaskGraph;
  private:

    Task* task;
    const PatchSubset* patches;
    const MaterialSubset* matls;
    map<DependencyBatch*, DependencyBatch*> reqs;
    DependencyBatch* comp_head;
    DetailedTasks* taskGroup;

    // Called when prerequisite tasks (dependencies) call done.
    void dependencySatisfied(InternalDependency* dep);

    
    // Internal dependencies are dependencies within the same process.
    list<InternalDependency> internalDependencies;
    
    // internalDependents will point to InternalDependency's in the
    // internalDependencies list of the requiring DetailedTasks.
    map<DetailedTask*, InternalDependency*> internalDependents;
    
    unsigned long numPendingInternalDependencies;
    Mutex internalDependencyLock;
    
    ScrubItem* scrublist;
    int resourceIndex;

    DetailedTask(const Task&);
    DetailedTask& operator=(const Task&);
  };

  class DetailedTasks {
  public:
    DetailedTasks(const ProcessorGroup* pg, const TaskGraph* taskgraph,
		  bool mustConsiderInternalDependencies = false);
    ~DetailedTasks();

    void add(DetailedTask* task);
    int numTasks() const {
      return (int)tasks.size();
    }
    DetailedTask* getTask(int i) {
      return tasks[i];
    }

    void assignMessageTags();
    int getMaxMessageTag() const {
      return maxSerial;
    }

    void possiblyCreateDependency(DetailedTask* from, Task::Dependency* comp,
				  const Patch* fromPatch,
				  DetailedTask* to, Task::Dependency* req,
				  const Patch* toPatch, int matl,
				  const IntVector& low, const IntVector& high);

    DetailedTask* getOldDWSendTask(int proc);

#if 0
    vector<DetailedReq*>& getInitialRequires();
#endif

    void initTimestep();
    
    void computeLocalTasks(int me);
    int numLocalTasks() const {
      return (int)localtasks.size();
    }
    
    DetailedTask* localTask(int idx) {
      return localtasks[idx];
    }

    DetailedTask* getNextInternalReadyTask();
    
    void createScrublists(bool init_timestep);

    bool mustConsiderInternalDependencies()
    { return mustConsiderInternalDependencies_; }

    unsigned long getCurrentDependencyGeneration()
    { return currentDependencyGeneration_; }
  protected:
    friend class DetailedTask;

    void internalDependenciesSatisfied(DetailedTask* task);
  private:
    void initializeBatches();

    void incrementDependencyGeneration()
    {
      if (currentDependencyGeneration_ >= ULONG_MAX)
	throw InternalError("DetailedTasks::currentDependencySatisfyingGeneration has overflowed");
      currentDependencyGeneration_++;
    }
    
    vector<DetailedTask*> tasks;
#if 0
    vector<DetailedReq*> initreqs;
#endif
    const TaskGraph* taskgraph;
    vector<Task*> stasks;
    vector<DetailedTask*> localtasks;
    vector<DependencyBatch*> batches;
    DetailedDep* initreq;
    
    int maxSerial;

    // True for mixed scheduler which needs to keep track of internal
    // depedencies.
    bool mustConsiderInternalDependencies_;

    list<DetailedTask*> internalDependencySatisfiedTasks_;
    list<DetailedTask*> initiallyReadyTasks_;

    // This "generation" number is to keep track of which InternalDependency
    // links have been satisfied in the current timestep and avoids the
    // need to traverse all InternalDependency links to reset values.
    unsigned long currentDependencyGeneration_;
    Mutex readyQueueMutex_;
    Semaphore readyQueueSemaphore_;

    DetailedTasks(const DetailedTasks&);
    DetailedTasks& operator=(const DetailedTasks&);
  };
} // End namespace Uintah

ostream& operator<<(ostream& out, const Uintah::DetailedTask& task);
ostream& operator<<(ostream& out, const Uintah::DetailedDep& task);

#endif

