
#ifndef UINTAH_HOMEBREW_Output_H
#define UINTAH_HOMEBREW_Output_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/OutputP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>

class Output : public UintahParallelPort {
public:
    Output();
    virtual ~Output();

    void finalizeTimestep(double t, double delt, const LevelP&, SchedulerP&,
			  const DataWarehouseP&);
private:
    Output(const Output&);
    Output& operator=(const Output&);
};

#endif
