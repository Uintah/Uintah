
#include "Task.h"
#include "SchedulerException.h"
#include <iostream>
using std::cerr;
using std::vector;

Task::ActionBase::~ActionBase()
{
}

Task::~Task()
{
    for(vector<Dependency*>::iterator iter=reqs.begin();
	iter != reqs.end(); iter++)
	delete *iter;
    for(vector<Dependency*>::iterator iter=comps.begin();
	iter != comps.end(); iter++)
	delete *iter;

    delete action;
}

void Task::usesMPI(bool state)
{
    d_usesMPI = state;
}

void Task::usesThreads(bool state)
{
    d_usesThreads = state;
}

void Task::subregionCapable(bool state)
{
    d_subregionCapable = state;
}

void Task::requires(const DataWarehouseP& ds, const std::string& name,
		    const TypeDescription* td)
{
    reqs.push_back(new Dependency(this, ds, name, td, 0, 0));
}

void Task::requires(const DataWarehouseP& ds, const std::string& name,
		    const Region* region, int numGhostCells,
		    const TypeDescription* td)
{
    reqs.push_back(new Dependency(this, ds, name, td, region, numGhostCells));
}

void Task::computes(const DataWarehouseP& ds, const std::string& name,
		    const TypeDescription* td)
{
    comps.push_back(new Dependency(this, ds, name, td, 0, 0));
}

void Task::computes(const DataWarehouseP& ds, const std::string& name,
		    const Region*, int numGhostCells,
		    const TypeDescription* td)
{
    comps.push_back(new Dependency(this, ds, name, td, region, numGhostCells));
}

void Task::doit(const ProcessorContext* pc)
{
    if(completed)
	throw SchedulerException("Task performed, but already completed");
    action->doit(pc, region, fromDW, toDW);
    //completed=true;
}

Task::Dependency::Dependency(Task* task, const DataWarehouseP& dw,
			     std::string varname,
			     const TypeDescription* vartype,
			     const Region* region, int numGhostCells)
    : task(task), dw(dw), varname(varname), vartype(vartype), region(region),
      numGhostCells(numGhostCells)
{
}

void Task::addReqs(std::vector<Dependency*>& to) const
{
    for(vector<Dependency*>::const_iterator iter = reqs.begin();
	iter != reqs.end(); iter++)
	to.push_back(*iter);
}

void Task::addComps(std::vector<Dependency*>& to) const
{
    for(vector<Dependency*>::const_iterator iter = comps.begin();
	iter != comps.end(); iter++)
	to.push_back(*iter);
}
