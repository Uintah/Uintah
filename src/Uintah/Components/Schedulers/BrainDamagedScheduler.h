
#ifndef UINTAH_HOMEBREW_BRAINDAMAGEDMAPPER_H
#define UINTAH_HOMEBREW_BRAINDAMAGEDMAPPER_H

#include "Scheduler.h"
#include <vector>

namespace SCICore {
    namespace Thread {
	class SimpleReducer;
	class ThreadPool;
    }
}

class BrainDamagedScheduler : public Scheduler {
public:
    BrainDamagedScheduler();
    virtual ~BrainDamagedScheduler();

    virtual void initialize();
    virtual void execute(const ProcessorContext*);
    virtual void addTarget(const std::string&);
    virtual void addTask(Task* t);
    virtual DataWarehouseP createDataWarehouse();
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

    BrainDamagedScheduler(const BrainDamagedScheduler&);
    BrainDamagedScheduler& operator=(const BrainDamagedScheduler&);
};

#endif
