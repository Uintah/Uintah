#ifndef UINTAH_HOMEBREW_DetailedTasks_H
#define UINTAH_HOMEBREW_DetailedTasks_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <vector>

namespace Uintah {
  class ProcessorGroup;
  class DataWarehouse;
  class DetailedTask;

  class DetailedDep {
  public:
    DetailedDep(DetailedDep* next, Task::Dependency* comp,
		Task::Dependency* req, const Patch* fromPatch,
		int matl,
		const IntVector& low, const IntVector& high)
      : next(next), comp(comp), req(req), fromPatch(fromPatch),
	low(low), high(high), matl(matl)
    {
    }
    DetailedDep* next;
    Task::Dependency* comp;
    Task::Dependency* req;
    const Patch* fromPatch;
    IntVector low, high;
    int matl;
  };

  class DependencyBatch {
  public:
    DependencyBatch(int to, DetailedTask* fromTask, DetailedTask* toTask)
      : req_next(0), comp_next(0), fromTask(fromTask), toTask(toTask),
	head(0), messageTag(-1), to(to)
    {
    }
    ~DependencyBatch();

    DependencyBatch* req_next;
    DependencyBatch* comp_next;
    DetailedTask* fromTask;
    DetailedTask* toTask;
    DetailedDep* head;
    int messageTag;
    int to;

  private:

    DependencyBatch(const DependencyBatch&);
    DependencyBatch& operator=(const DependencyBatch&);
  };

  class DetailedTask {
  public:
    DetailedTask(Task* task, const PatchSubset* patches,
		 const MaterialSubset* matls);
    ~DetailedTask();
    
    void doit(const ProcessorGroup* pg, DataWarehouse* old_dw,
	      DataWarehouse* new_dw);

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

    DependencyBatch* getRequires() {
      return req_head;
    }
    DependencyBatch* getComputes() {
      return comp_head;
    }

    void addRequires(DependencyBatch*);
    void addComputes(DependencyBatch*);

  protected:
    friend class TaskGraph;
  private:

    Task* task;
    const PatchSubset* patches;
    const MaterialSubset* matls;
    DependencyBatch* req_head;
    DependencyBatch* comp_head;
    int resourceIndex;

    DetailedTask(const Task&);
    DetailedTask& operator=(const Task&);
  };

  class DetailedTasks {
  public:
    DetailedTasks(const ProcessorGroup* pg);
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

    void computeLocalTasks(int me);
    int numLocalTasks() const {
      return (int)localtasks.size();
    }
    DetailedTask* localTask(int idx) {
      return localtasks[idx];
    }
  private:
    vector<DetailedTask*> tasks;
#if 0
    vector<DetailedReq*> initreqs;
#endif
    vector<Task*> stasks;
    vector<DetailedTask*> localtasks;
    vector<DependencyBatch*> batches;
    DetailedDep* initreq;
    
    int maxSerial;

    DetailedTasks(const DetailedTasks&);
    DetailedTasks& operator=(const DetailedTasks&);
  };
} // End namespace Uintah

ostream& operator<<(ostream& out, const Uintah::DetailedTask& task);
ostream& operator<<(ostream& out, const Uintah::DetailedDep& task);

#endif
