/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Task.h>
#include <Uintah/Exceptions/SchedulerException.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <iostream>

namespace Uintah {
namespace Grid {

Task::ActionBase::~ActionBase()
{
}

Task::~Task()
{
  vector<Dependency*>::iterator iter;

  for( iter=d_reqs.begin(); iter != d_reqs.end(); iter++ )
    { delete *iter; }
  for( iter=d_comps.begin(); iter != d_comps.end(); iter++)
    { delete *iter; }
  delete d_action;
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
  d_reqs.push_back(new Dependency(this, ds, name, td, 0, 0));
}

void Task::requires(const DataWarehouseP& ds, const std::string& name,
		    const Region* region, int numGhostCells,
		    const TypeDescription* td)
{
  d_reqs.push_back(new Dependency(this, ds, name, td, region, numGhostCells));
}

void Task::computes(const DataWarehouseP& ds, const std::string& name,
		    const TypeDescription* td)
{
  d_comps.push_back(new Dependency(this, ds, name, td, 0, 0));
}

void Task::computes(const DataWarehouseP& ds, const std::string& name,
		    const Region*, int numGhostCells,
		    const TypeDescription* td)
{
  d_comps.push_back(new Dependency(this, ds, name, td, 
				   d_region, numGhostCells));
}

void Task::doit(const ProcessorContext* pc)
{
  if( d_completed )
    throw SchedulerException("Task performed, but already completed");
  d_action->doit(pc, d_region, d_fromDW, d_toDW);
  //d_completed=true;
}

Task::Dependency::Dependency(Task* task, const DataWarehouseP& dw,
			     std::string varname,
			     const TypeDescription* vartype,
			     const Region* region, int numGhostCells)
    : d_task(task),
      d_dw(dw),
      d_varname(varname),
      d_vartype(vartype),
      d_region(region),
      d_numGhostCells(numGhostCells)
{
}

void Task::addReqs(std::vector<Dependency*>& to) const
{
  vector<Dependency*>::const_iterator iter;

  for( iter = d_reqs.begin(); iter != d_reqs.end(); iter++ )
    {
      to.push_back(*iter);
    }
}

void Task::addComps(std::vector<Dependency*>& to) const
{
  vector<Dependency*>::const_iterator iter;

  for( iter = d_comps.begin(); iter != d_comps.end(); iter++ )
    {
      to.push_back(*iter);
    }
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/03/17 09:29:59  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
