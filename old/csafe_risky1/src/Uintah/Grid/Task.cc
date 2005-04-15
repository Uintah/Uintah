//
// $Id$
//

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
  d_reqs.push_back(Dependency(ds.get_rep(), var, -1, 0, this,
			      IntVector(-9,-8,-7), IntVector(-6,-5,-4)));
}

void
Task::requires(const DataWarehouseP& ds, const VarLabel* var, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype, int numGhostCells)
{
   ASSERT(ds.get_rep() != 0);
   const TypeDescription* td = var->typeDescription();
   Level::selectType neighbors;
   IntVector lowIndex, highIndex;
   patch->computeVariableExtents(td->getType(), gtype, numGhostCells,
				 neighbors, lowIndex, highIndex);
   switch ( td->getType()) {
   case TypeDescription::CCVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getCellLowIndex());
      IntVector high= Min(highIndex, neighbor->getCellHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCXVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getSFCXLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCXHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCYVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getSFCYLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCYHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCZVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getSFCZLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCZHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
     break;
   case TypeDescription::NCVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
      IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
     break;
   default:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      using SCICore::Geometry::Max;
      using SCICore::Geometry::Min;
      IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
      IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
   break;
   }
}


void
Task::computes(const DataWarehouseP& ds, const VarLabel* var)
{
   ASSERT(ds.get_rep() != 0);
   d_comps.push_back(Dependency(ds.get_rep(), var, -1, d_patch, this,
				IntVector(-19,-18,-17),
				IntVector(-16,-15,-14)));
}

void
Task::computes(const DataWarehouseP& ds, const VarLabel* var, int matlIndex,
	       const Patch* patch)
{
   ASSERT(ds.get_rep() != 0);
   d_comps.push_back(Dependency(ds.get_rep(), var, matlIndex, patch,
				this,
				IntVector(-29,-28,-27),
				IntVector(-26,-25,-24)));
}

void
Task::doit(const ProcessorGroup* pc)
{
  if( d_completed )
      throw InternalError("Task doit() called, but has already completed");
  if(d_action)
     d_action->doit(pc, d_patch, d_fromDW, d_toDW);
  d_completed=true;
}

void
Task::display( ostream & out ) const
{
  out << getName() << " (" << d_tasktype << "): [Own: " << d_resourceIndex
      << ", ";
  if( d_patch != 0 ){
    out << "P: " << d_patch->getID()
	<< ", DW: " << d_fromDW->getID()
	<< ", " << d_toDW->getID() << ", ";
  } else {
    out << "(No Patch), ";
  }
  if( d_completed ){ out << "Completed, "; } else { out << "Pending, "; }
  out << "(R: " << d_reqs.size() << ", C: " << d_comps.size() << ")]";
}

ostream &
operator << ( ostream & out, const Uintah::Task::Dependency & dep )
{
  out << "[" << *(dep.d_var) << " Patch: ";
  if( dep.d_patch ){
    out << dep.d_patch->getID();
  } else {
    out << "none";
  }
  out << " MI: " << dep.d_matlIndex << " DW: " << dep.d_dw->getID() << " SN: " 
      << dep.d_serialNumber << "]";
  return out;
}

void
Task::displayAll(ostream& out) const
{
   display(out);
   out << '\n';
   for(int i=0;i<(int)d_reqs.size();i++)
      out << "requires: " << d_reqs[i] << '\n';
   for(int i=0;i<(int)d_comps.size();i++)
      out << "computes: " << d_comps[i] << '\n';
}

ostream &
operator << (ostream &out, const Task & task)
{
  task.display( out );
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
// Revision 1.24.2.3  2000/10/20 02:06:37  rawat
// modified cell centered and staggered variables to optimize communication
//
// Revision 1.24.2.2  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.24.2.1  2000/09/29 06:12:29  sparker
// Added support for sending data along patch edges
//
// Revision 1.24  2000/09/28 23:22:01  jas
// Added (int) to remove g++ warnings for STL size().  Reordered initialization
// to coincide with *.h declarations.
//
// Revision 1.23  2000/09/26 21:38:36  dav
// minor updates
//
// Revision 1.22  2000/09/25 17:30:07  sparker
// Use correct patch for computes
//
// Revision 1.21  2000/09/25 16:24:17  sparker
// Added a displayAll method to Task
//
// Revision 1.20  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
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
