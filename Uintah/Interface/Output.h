
#ifndef UINTAH_HOMEBREW_Output_H
#define UINTAH_HOMEBREW_Output_H

#include "OutputP.h"
#include "DataWarehouseP.h"
#include "GridP.h"
#include "Handle.h"
#include "LevelP.h"
#include "SchedulerP.h"
#include "RefCounted.h"

class Output : public RefCounted {
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
