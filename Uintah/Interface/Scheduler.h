
#ifndef UINTAH_HOMEBREW_MAPPER_H
#define UINTAH_HOMEBREW_MAPPER_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <string>

class ProcessorContext;
class Task;

class Scheduler : public UintahParallelPort {
public:
    Scheduler();
    virtual ~Scheduler();

    virtual void initialize() = 0;
    virtual void execute(const ProcessorContext*) = 0;
    virtual void addTarget(const std::string& name) = 0;
    virtual void addTask(Task* t) = 0;
    virtual DataWarehouseP createDataWarehouse() = 0;
private:
    Scheduler(const Scheduler&);
    Scheduler& operator=(const Scheduler&);
};

#endif
