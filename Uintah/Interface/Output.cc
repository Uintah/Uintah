
#include "Output.h"
#include <iostream>
using std::cerr;

Output::Output()
{
}

Output::~Output()
{
}

void Output::finalizeTimestep(double t, double, const LevelP&,
			      SchedulerP&, const DataWarehouseP&)
{
    //cerr << "Finalizing time step: t=" << t << '\n';
}

