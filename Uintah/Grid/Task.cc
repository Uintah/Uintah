/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <iostream>

using namespace Uintah;
using SCICore::Exceptions::InternalError;

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

void Task::requires(const DataWarehouseP& ds, const VarLabel* var)
{
  d_reqs.push_back(new Dependency(this, ds, var, 0, 0));
}

void Task::requires(const DataWarehouseP& ds, const VarLabel* var,
		    const Region* region, int numGhostCells)
{
  d_reqs.push_back(new Dependency(this, ds, var, region, numGhostCells));
}

void Task::computes(const DataWarehouseP& ds, const VarLabel* var)
{
  d_comps.push_back(new Dependency(this, ds, var, 0, 0));
}

void Task::computes(const DataWarehouseP& ds, const VarLabel* var,
		    const Region*)
{
  d_comps.push_back(new Dependency(this, ds, var,
				   d_region, 0));
}

void Task::doit(const ProcessorContext* pc)
{
  if( d_completed )
      throw InternalError("Task performed, but already completed");
  d_action->doit(pc, d_region, d_fromDW, d_toDW);
  //d_completed=true;
}

Task::Dependency::Dependency(Task* task, const DataWarehouseP& dw,
			     const VarLabel* var,
			     const Region* region, int numGhostCells)
    : d_task(task),
      d_dw(dw),
      d_var(var),
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

//
// $Log$
// Revision 1.7  2000/04/26 06:49:00  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/20 18:56:31  sparker
// Updates to MPM
//
// Revision 1.5  2000/04/11 07:10:50  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/17 09:29:59  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
