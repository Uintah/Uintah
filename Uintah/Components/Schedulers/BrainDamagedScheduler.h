
#ifndef UINTAH_HOMEBREW_BRAINDAMAGEDScheduler_H
#define UINTAH_HOMEBREW_BRAINDAMAGEDScheduler_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/Scheduler.h>
#include <vector>

namespace SCICore {
    namespace Thread {
	class SimpleReducer;
	class ThreadPool;
    }
}

class BrainDamagedScheduler : public UintahParallelComponent, public Scheduler {
public:
    BrainDamagedScheduler();
    virtual ~BrainDamagedScheduler();

    virtual void initialize();
    virtual void execute(const ProcessorContext*);
    virtual void addTarget(const std::string&);
    virtual void addTask(Task* t);
    virtual DataWarehouseP createDataWarehouse();
    void setNumThreads(int numThreads);
private:
    SCICore::Thread::SimpleReducer* reducer;
    struct TaskRecord {
	Task* task;
	std::vector<TaskRecord*> deps;
	std::vector<TaskRecord*> reverseDeps;
	TaskRecord(Task*);
	~TaskRecord();
    };
    bool allDependenciesCompleted(TaskRecord* task) const;
    void setupTaskConnections();

    void runThreadedTask(int, TaskRecord*, const ProcessorContext*,
			 SCICore::Thread::SimpleReducer*);

    std::vector<TaskRecord*> tasks;
    std::vector<std::string> targets;

    SCICore::Thread::ThreadPool* pool;
    int numThreads;

    BrainDamagedScheduler(const BrainDamagedScheduler&);
    BrainDamagedScheduler& operator=(const BrainDamagedScheduler&);
};

#endif
