
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;

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
Task::requires(const DataWarehouseP& ds, const VarLabel* var,
	       int matlIndex /* = -1 */)
{
  d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex, 0, this,
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
      IntVector low = Max(lowIndex, neighbor->getCellLowIndex());
      IntVector high= Min(highIndex, neighbor->getCellHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCXVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector low = Max(lowIndex, neighbor->getSFCXLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCXHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCYVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector low = Max(lowIndex, neighbor->getSFCYLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCYHighIndex());
      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
   }
     break;
   case TypeDescription::SFCZVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector low = Max(lowIndex, neighbor->getSFCZLowIndex());
      IntVector high= Min(highIndex, neighbor->getSFCZHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
     break;
   case TypeDescription::NCVariable:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
      IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
     break;
   default:
     for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
      IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

      d_reqs.push_back(Dependency(ds.get_rep(), var, matlIndex,
				  neighbor, this, low, high));
     }
   break;
   }
}


void
Task::computes(const DataWarehouseP& ds, const VarLabel* var,
	       int matlIndex /* = -1 */)
{
   ASSERT(ds.get_rep() != 0);
   d_comps.push_back(Dependency(ds.get_rep(), var, matlIndex, d_patch, this,
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

