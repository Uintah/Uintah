/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Malloc/Allocator.h>
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

void
Task::usesMPI(bool state)
{
  d_usesMPI = state;
}

void
Task::usesThreads(bool state)
{
  d_usesThreads = state;
}

void
Task::subpatchCapable(bool state)
{
  d_subpatchCapable = state;
}

void
Task::requires(const DataWarehouseP& ds, const VarLabel* var)
{
  d_reqs.push_back(scinew Dependency(ds, var, -1, 0, this));
}

void
Task::requires(const DataWarehouseP& ds, const VarLabel* var, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype, int numGhostCells)
{
   ASSERT(ds.get_rep() != 0);
   const TypeDescription* td = var->typeDescription();
   int l,h;
   switch(gtype){
   case Ghost::None:
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      l=h=0;
      d_reqs.push_back(scinew Dependency(ds, var, matlIndex, patch, this));
      return;
   case Ghost::AroundNodes:
      if(numGhostCells == 0)
	 throw InternalError("No ghost cells specified with Task::AroundNodes");
      switch(td->getType()){
      case TypeDescription::NCVariable:
	 // All 27 neighbors
	 l=-1;
	 h=1;
	 break;
      case TypeDescription::CCVariable:
      case TypeDescription::ParticleVariable:
	 // Lower neighbors
	 l=-1;
	 h=0;
         break;
      default:
	 throw InternalError("Illegal Basis type");
      }
      break;
   case Ghost::AroundCells:
      if(numGhostCells == 0)
	 throw InternalError("No ghost cells specified with Task::AroundCells");
      switch(td->getType()){
      case TypeDescription::NCVariable:
	 // Upper neighbors
	 l=0;
	 h=1;
         break;
      case TypeDescription::CCVariable:
      case TypeDescription::ParticleVariable:
	 // All 27 neighbors
	 l=-1;
	 h=1;
	 break;
      default:
	 throw InternalError("Illegal Basis type");
      }
      break;
   default:
      throw InternalError("Illegal ghost type");
   }
   const Level* level = patch->getLevel();
   std::vector<const Patch*> neighbors;
   IntVector low(patch->getCellLowIndex()+IntVector(l,l,l));
   IntVector high(patch->getCellHighIndex()+IntVector(h,h,h));
   level->selectPatches(low, high, neighbors);
   for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      d_reqs.push_back(scinew Dependency(ds, var, matlIndex,
					 neighbor, this));
   }
}

void
Task::computes(const DataWarehouseP& ds, const VarLabel* var)
{
   ASSERT(ds.get_rep() != 0);
   d_comps.push_back(scinew Dependency(ds, var, -1, d_patch, this));
}

void
Task::computes(const DataWarehouseP& ds, const VarLabel* var, int matlIndex,
	       const Patch*)
{
   ASSERT(ds.get_rep() != 0);
   d_comps.push_back(scinew Dependency(ds, var, matlIndex, d_patch, this));
}

void
Task::doit(const ProcessorGroup* pc)
{
  if( d_completed )
      throw InternalError("Task performed, but already completed");
  if(d_action)
     d_action->doit(pc, d_patch, d_fromDW, d_toDW);
  d_completed=true;
}

Task::Dependency::Dependency(const DataWarehouseP& dw,
			     const VarLabel* var, int matlIndex,
			     const Patch* patch,
			     Task* task)
    : d_dw(dw),
      d_var(var),
      d_matlIndex(matlIndex),
      d_patch(patch),
   d_task(task),
   d_serialNumber(-123)
{
}

const vector<Task::Dependency*>&
Task::getComputes() const
{
  return d_comps;
}

const vector<Task::Dependency*>&
Task::getRequires() const
{
  return d_reqs;
}

void
Task::display( ostream & out ) const
{
  out << d_taskName << " (" << d_tasktype << "): ";
  if( d_patch != 0 ){
    out << "[Patch: " << d_patch->getID() << ", DW: " << d_fromDW->getID()
	<< ", " << d_toDW->getID() << ", ";
  } else {
    out << "(No Patch), ";
  }
  if( d_completed ){ out << "Completed]"; } else { out << "Not Completed]"; }
}

ostream &
operator << (ostream &out, const Task & task)
{
  task.display( out );
  return out;
}

ostream&
operator << (ostream& out, const Task::Dependency& dep)
{
   out << "[dep: var=" << dep.d_var->getName();
   if(dep.d_patch)
      out << ", patch=" << dep.d_patch->getID();
   out << ", matl=" << dep.d_matlIndex << ", dw=" << dep.d_dw->getID() << " ]";
   return out;
}

ostream &
operator << (ostream &out, const Task::TaskType & tt)
{
  switch( tt ) {
  case Task::Normal:
    out << "Normal";
    break;
  case Task::Reduction:
    out << "Reduction";
    break;
  case Task::Scatter:
    out << "Scatter";
    break;
  case Task::Gather:
    out << "Gather";
    break;
  }
  return out;
}

//
// $Log$
// Revision 1.19  2000/09/13 20:57:25  sparker
// Added ostream operator for dependencies
//
// Revision 1.18  2000/09/12 15:11:35  sparker
// Added assertions to ensure we have a valid data warehouse for computes/requires
//
// Revision 1.17  2000/08/23 22:33:40  dav
// added an output operator for task
//
// Revision 1.16  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.15  2000/06/17 07:06:44  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.14  2000/06/15 21:57:19  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.13  2000/06/03 05:29:45  sparker
// Changed reduction variable emit to require ostream instead of ofstream
// emit now only prints number without formatting
// Cleaned up a few extraneously included files
// Added task constructor for an non-patch-based action with 1 argument
// Allow for patches and actions to be null
// Removed back pointer to this from Task::Dependency
//
// Revision 1.12  2000/05/30 20:19:34  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.11  2000/05/20 08:09:27  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.10  2000/05/10 20:03:03  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.9  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.8  2000/05/05 06:42:45  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
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
