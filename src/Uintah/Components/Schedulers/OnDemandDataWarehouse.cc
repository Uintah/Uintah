/* REFERENCED */
static char *id="@(#) $Id$";

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Guard.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/IntVector.h>

#include <Uintah/Interface/DWMpiHandler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Exceptions/UnknownVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream>
#include <string>
#include <mpi.h>

using std::cerr;
using std::string;
using std::vector;

using SCICore::Exceptions::InternalError;
using SCICore::Thread::Guard;
using SCICore::Geometry::Point;

namespace Uintah {

OnDemandDataWarehouse::OnDemandDataWarehouse( int MpiRank, 
					      int MpiProcesses,
					      int generation ) :
  d_lock("DataWarehouse lock"),
  DataWarehouse( MpiRank, MpiProcesses, generation ),
  d_responseTag( 0 )
{
  d_finalized = false;
  d_positionLabel = scinew VarLabel("__internal datawarehouse position variable",
				ParticleVariable<Point>::getTypeDescription(),
				VarLabel::Internal);
}

void
OnDemandDataWarehouse::setGrid(const GridP& grid)
{
  d_grid = grid;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
}

void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
			   const VarLabel* label)
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);

   if(iter == d_reductionDB.end()) {
      cerr << "OnDemandDataWarehouse: get Reduction: UnknownVariable: " 
	   << label->getName() << "\n";
      throw UnknownVariable(label->getName());
   }

   var.copyPointer(*iter->second->var);
}

void
OnDemandDataWarehouse::allocate(ReductionVariableBase&,
				const VarLabel*)
{
   cerr << "OnDemend DataWarehouse::allocate(ReductionVariable) "
	<< "not finished\n";
}

void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
			   const VarLabel* label)
{
   ASSERT(!d_finalized);

   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter == d_reductionDB.end()){
      d_reductionDB[label]=scinew ReductionRecord(var.clone());
      iter = d_reductionDB.find(label);
   } else {
      iter->second->var->reduce(var);
   }
}

void
OnDemandDataWarehouse::sendMpiDataRequest( const string & varName,
					         Patch * patch,
					         int      numGhostCells )
{
  if( d_MpiProcesses == 1 ) {
      throw InternalError( "sendMpiDataRequest should not be called if"
			   " there is only one process" );
  }

  // Must send a reqest to 26 neighbors:
  // 0-8 are above, 8-17 are on the same level (this node is 13), 
  // 18-26 are below.
  //
  //                                                      0  1  2
  //                                 /  /  /              /\  \  \
  //                                9 10 11                 3  4  5
  //          /  /  /              / \  \  \                 \  \  \
  //         18 19 20                12 13 14                 6  7  8
  //          \  \  \                  \  \  \/               /  /  /
  //          21 22 23                 15 16 17
  //            \  \  \/               /  /  /
  //            24 25 26
  //
  //  8 is directly above 17 which is directly above 26


  // Figure out the bottom and top points of the ghost patch
  // immediately above this patch.  (Ie: Determine Top (Area 4))

  int                           currentTag;
  DWMpiHandler::MpiDataRequest  request;

  request.fromMpiRank = d_MpiRank;
  request.toMpiRank = !d_MpiRank; // Just a testing hack...

  d_lock.writeLock();
  request.tag = d_responseTag;
  currentTag = d_responseTag;
  d_responseTag++;
  d_lock.writeUnlock();

  request.type = DWMpiHandler::GridVar;
  sprintf( request.varName, "varA" );
  request.patch = 0;
  request.generation = d_generation;

  cerr << "OnDemandDataWarehouse " << d_MpiRank << ": sending data request\n";

  MPI_Bsend( (void*)&request, sizeof( request ), MPI_BYTE, request.toMpiRank,
	     DWMpiHandler::DATA_REQUEST_TAG, MPI_COMM_WORLD );

  cerr << "                       Data request sent\n";

  char       * buffer = new char[ DWMpiHandler::MAX_BUFFER_SIZE ];
  MPI_Status   status;

  cerr << "OnDemandDataWarehouse: waiting for data response from "
       << request.toMpiRank << "\n";

  MPI_Recv( buffer, 100, MPI_BYTE, request.toMpiRank, 
	    currentTag, MPI_COMM_WORLD, &status );

    cerr << "STATUS IS:\n";
    cerr << "SOURCE: " << status.MPI_SOURCE << "\n";
    cerr << "TAG:    " << status.MPI_TAG << "\n";
    cerr << "ERROR:  " << status.MPI_ERROR << "\n";
    cerr << "SIZE:   " << status.size << "\n";

  cerr << "Received this message: [" << buffer << "]\n";
  free( buffer );
}

void
OnDemandDataWarehouse::get(ParticleVariableBase& var,
			   const VarLabel* label,
			   int matlIndex,
			   const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
   if(!d_particleDB.exists(label, matlIndex, patch))
      throw UnknownVariable(label->getName());
#if 1
   if(gtype == Ghost::None){
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      d_particleDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      switch(gtype){
      case Ghost::AroundNodes:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundNodes");
	 // Lower neighbors
	 l=-1;
	 h=0;
	 break;
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // All 27 neighbors
	 l=-1;
	 h=1;
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      Box box = patch->getGhostBox(l, h);
      particleIndex totalParticles = 0;
      vector<ParticleVariableBase*> neighborvars;
      vector<ParticleSubset*> subsets;
      for(int ix=l;ix<=h;ix++){
	 for(int iy=l;iy<=h;iy++){
	    for(int iz=l;iz<=h;iz++){
	       const Patch* neighbor = patch->getNeighbor(IntVector(ix,iy,iz));
	       if(neighbor){
		  if(!d_particleDB.exists(d_positionLabel,matlIndex,neighbor))
		     throw InternalError("Position variable does not exist: "+ 
			  d_positionLabel->getName());
		  if(!d_particleDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
			  d_positionLabel->getName());

		  ParticleVariable<Point> pos;
		  d_particleDB.get(d_positionLabel, matlIndex, neighbor, pos);
		  ParticleSubset* pset = pos.getParticleSubset();
		  ParticleSubset* subset = 
                        scinew ParticleSubset(pset->getParticleSet(), false);
		  for(ParticleSubset::iterator iter = pset->begin();
		      iter != pset->end(); iter++){
		     particleIndex idx = *iter;
		     if(box.contains(pos[idx])){
			subset->addParticle(idx);
		     }
		  }
		  totalParticles+=subset->numParticles();
		  neighborvars.push_back(
                                 d_particleDB.get(label, matlIndex, neighbor));
		  subsets.push_back(subset);
	       }
	    }
	 }
      }
      ParticleSet* newset = scinew ParticleSet(totalParticles);
      ParticleSubset* newsubset = scinew ParticleSubset(newset, true);
      var.gather(newsubset, subsets, neighborvars);
      for(int i=0;i<subsets.size();i++)
	 delete subsets[i];
   }
#endif
}

void
OnDemandDataWarehouse::allocate(int numParticles,
				ParticleVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Particle variable already exists: "+label->getName());
   if(!label->isPositionVariable())
      throw InternalError("Particle allocate via numParticles should "
			  "only be used for position variables");
   if(d_particleDB.exists(d_positionLabel, matlIndex, patch))
      throw InternalError("Particle position already exists in datawarehouse");

   // Create the particle set and variable
   ParticleSet* pset = scinew ParticleSet(numParticles);
   ParticleSubset* psubset = scinew ParticleSubset(pset, true);
   var.allocate(psubset);

   // Put it in the database
   d_particleDB.put(d_positionLabel, matlIndex, patch, var, false);
}

void
OnDemandDataWarehouse::allocate(ParticleVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Particle variable already exists: " +
			  label->getName());
   if(!d_particleDB.exists(d_positionLabel, matlIndex, patch))
      throw InternalError("Position variable does not exist: " + 
			  d_positionLabel->getName());
   ParticleVariable<Point> pos;
   d_particleDB.get(d_positionLabel, matlIndex, patch, pos);

   // Allocate the variable
   var.allocate(pos.getParticleSubset());
}

void
OnDemandDataWarehouse::put(const ParticleVariableBase& var,
			   const VarLabel* label,
			   int matlIndex,
			   const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Variable already exists: "+label->getName());

   // Put it in the database
   d_particleDB.put(label, matlIndex, patch, var, true);
}

void
OnDemandDataWarehouse::get(NCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 1
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_ncDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName());
      d_ncDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundNodes");
	 // All 27 neighbors
	 l=-1;
	 h=1;
	 lowIndex = patch->getNodeLowIndex()-gc;
	 highIndex = patch->getNodeHighIndex()+gc;
	 cerr << "Nodes around nodes is probably not functional!\n";
	 break;
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 l=0;
	 h=1;
	 lowIndex = patch->getCellLowIndex();
         highIndex = patch->getCellHighIndex()+gc;
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalNodes=0;
      for(int ix=l;ix<=h;ix++){
	 for(int iy=l;iy<=h;iy++){
	    for(int iz=l;iz<=h;iz++){
	       const Patch* neighbor = patch->getNeighbor(IntVector(ix,iy,iz));
	       if(neighbor){
		  if(!d_ncDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
					 label->getName());
		  NCVariableBase* srcvar = 
		    d_ncDB.get(label, matlIndex, neighbor);

		  using SCICore::Geometry::Max;
		  using SCICore::Geometry::Min;

		  IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
		  IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

		  if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		      || ( high.z() < low.z() ) )
		     throw InternalError("Patch doesn't overlap?");

		  var.copyPatch(srcvar, low, high);
		  IntVector dnodes = high-low;
		  totalNodes+=dnodes.x()*dnodes.y()*dnodes.z();
	       }
	    }
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantnodes = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantnodes, totalNodes);
   }
#endif
}

void
OnDemandDataWarehouse::allocate(NCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_ncDB.exists(label, matlIndex, patch))
      throw InternalError("NC variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getNodeLowIndex(), patch->getNodeHighIndex());
}

void
OnDemandDataWarehouse::put(const NCVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_ncDB.exists(label, matlIndex, patch))
      throw InternalError("NC variable already exists: "+label->getName());

   // Put it in the database
   d_ncDB.put(label, matlIndex, patch, var, true);
}

void
OnDemandDataWarehouse::allocate(CCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_ccDB.exists(label, matlIndex, patch))
      throw InternalError("CC variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getCellLowIndex(), patch->getCellHighIndex());
}

void
OnDemandDataWarehouse::get(CCVariableBase& var, const VarLabel* label,
			   int matlIndex,
			   const Patch* patch, Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 1
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_ccDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName());
      d_ccDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundNodes");
	 // All 27 neighbors
	 l=-1;
	 h=1;
	 lowIndex = patch->getCellLowIndex()-gc;
	 highIndex = patch->getCellHighIndex()+gc;
	 cerr << "Cells around nodes is probably not functional!\n";
	 break;
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // all 6 faces
	 l=-1;
	 h=1;
	 lowIndex = patch->getCellLowIndex()-gc;
         highIndex = patch->getCellHighIndex()+gc;
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalCells=0;
      // change it to traverse only thru patches with adjoining faces
      for(int ix=l;ix<=h;ix++){
	 for(int iy=l;iy<=h;iy++){
	    for(int iz=l;iz<=h;iz++){
	       const Patch* neighbor = patch->getNeighbor(IntVector(ix,iy,iz));
	       if(neighbor){
		  if(!d_ccDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
					 label->getName());
		  CCVariableBase* srcvar = 
		    d_ccDB.get(label, matlIndex, neighbor);

		  using SCICore::Geometry::Max;
		  using SCICore::Geometry::Min;

		  IntVector low = Max(lowIndex, neighbor->getCellLowIndex());
		  IntVector high= Min(highIndex, neighbor->getCellHighIndex());

		  if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		      || ( high.z() < low.z() ) )
		     throw InternalError("Patch doesn't overlap?");

		  var.copyPatch(srcvar, low, high);
		  IntVector dcells = high-low;
		  totalCells+=dcells.x()*dcells.y()*dcells.z();
	       }
	    }
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantcells = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantcells, totalCells);
   }
#endif
}

void
OnDemandDataWarehouse::put(const CCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch )
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_ccDB.exists(label, matlIndex, patch))
      throw InternalError("CC variable already exists: "+label->getName());

   // Put it in the database
   d_ccDB.put(label, matlIndex, patch, var, true);
}

int
OnDemandDataWarehouse::findMpiNode( const VarLabel * label,
				    const Patch   * patch )
{
  variableListType * varList = d_dataLocation[ label ];

  char msg[ 1024 ];

  if( varList == 0 ) {
    sprintf( msg, "findMpiNode: Requested variable: %s for\n"
	     "patch %s is not in d_dataLocation",
	     label->getName().c_str(), patch->toString().c_str() );
    throw InternalError( string( msg ) );
  }
  // Run through all the different patches associated with "label" to
  // find which one contains the "patch" that has been requested.

  variableListType::iterator iter = varList->begin();

  while( iter != varList->end() ) {
    if( (*iter)->patch->contains( *patch ) ) {
      return (*iter)->mpiNode;
    }
    iter++;
  }

  sprintf( msg, "findMpiNode: Requested patch: %s for\n"
	   "patch %s is not in d_dataLocation",
	   label->getName().c_str(), patch->toString().c_str() );
  throw InternalError( string( msg ) );
}

void
OnDemandDataWarehouse::registerOwnership( const VarLabel * label,
					  const Patch   * patch,
					        int        mpiNode )
{
  variableListType * varList = d_dataLocation[ label ];

  if( varList == 0 ) {
    varList = scinew variableListType();
    d_dataLocation[ label ] = varList;
  }

  // Possibly should make sure that varList does not already have
  // this "label" with this "patch" in it...  for now assuming that 
  // this doesn't happen...

  dataLocation * location = scinew dataLocation();

  location->patch = patch;
  location->mpiNode = mpiNode;

  varList->push_back( location );
}

void
OnDemandDataWarehouse::carryForward(const DataWarehouseP& fromp)
{
   OnDemandDataWarehouse* from = dynamic_cast<OnDemandDataWarehouse*>(fromp.get_rep());

   d_grid = from->d_grid;

   for(int l = 0; l < d_grid->numLevels(); l++){
      const LevelP& level = d_grid->getLevel(l);
      for(Level::const_patchIterator iter = level->patchesBegin();
	  iter != level->patchesEnd(); iter++){
	 const Patch* patch = *iter;

	 d_particleDB.copyAll(from->d_particleDB, d_positionLabel, patch);
      }
   }
}

void
OnDemandDataWarehouse::pleaseSave(const VarLabel* var, int number)
{
   ASSERT(!d_finalized);

   d_saveset.push_back(var);
   d_savenumbers.push_back(number);
}

void
OnDemandDataWarehouse::pleaseSaveIntegrated(const VarLabel* var)
{
   ASSERT(!d_finalized);

   d_saveset_integrated.push_back(var);
}

void
OnDemandDataWarehouse::getSaveSet(std::vector<const VarLabel*>& vars,
				  std::vector<int>& numbers) const
{
   vars=d_saveset;
   numbers=d_savenumbers;
}

void
OnDemandDataWarehouse::getIntegratedSaveSet
				(std::vector<const VarLabel*>& vars) const
{
   vars=d_saveset_integrated;
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, const Patch* patch) const
{
   if(!patch){
      reductionDBtype::const_iterator iter = d_reductionDB.find(label);
      if(iter != d_reductionDB.end())
	 return true;
   } else {
      if(d_ncDB.exists(label, patch))
	 return true;
      if(d_particleDB.exists(label, patch))
	 return true;
   }
   return false;
}


void OnDemandDataWarehouse::emit(OutputContext& oc, const VarLabel* label,
				 int matlIndex, const Patch* patch) const
{
   if(d_ncDB.exists(label, matlIndex, patch)) {
      NCVariableBase* var = d_ncDB.get(label, matlIndex, patch);
      var->emit(oc);
      return;
   }
   if(d_particleDB.exists(label, matlIndex, patch)) {
      ParticleVariableBase* var = d_particleDB.get(label, matlIndex, patch);
      var->emit(oc);
      return;
   }
   throw UnknownVariable(label->getName());
}

void OnDemandDataWarehouse::emit(ofstream& intout,
				 vector <const VarLabel*> ivars) const
{

  static ts = 0;
  if(ts == 0){
     intout << "Step_number" << " ";
     for(int i = 0;i<ivars.size();i++){
	intout <<  ivars[i]->getName() << " ";
     }
     intout << endl;
  }

  for(int i = 0;i<ivars.size();i++){
//	ivars[i]->emit(intout);
  }
  intout << endl;

  ts++;
}

OnDemandDataWarehouse::ReductionRecord::ReductionRecord(ReductionVariableBase* var)
   : var(var)
{
}

} // end namespace Uintah

//
// $Log$
// Revision 1.28  2000/06/01 23:14:04  guilkey
// Added pleaseSaveIntegrated functionality to save ReductionVariables
// to an archive.
//
// Revision 1.27  2000/05/31 04:01:46  rawat
// partially completed CCVariable implementation
//
// Revision 1.26  2000/05/30 20:19:23  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.25  2000/05/30 17:09:37  dav
// MPI stuff
//
// Revision 1.24  2000/05/21 20:10:49  sparker
// Fixed memory leak
// Added scinew to help trace down memory leak
// Commented out ghost cell logic to speed up code until the gc stuff
//    actually works
//
// Revision 1.23  2000/05/15 20:04:35  dav
// Mpi Stuff
//
// Revision 1.22  2000/05/15 19:39:43  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.21  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.20  2000/05/10 20:02:53  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.19  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.18  2000/05/06 03:54:10  sparker
// Fixed multi-material carryForward
//
// Revision 1.17  2000/05/05 06:42:43  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.16  2000/05/02 17:54:29  sparker
// Implemented more of SerialMPM
//
// Revision 1.15  2000/05/02 06:07:16  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.14  2000/05/01 16:18:16  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.13  2000/04/28 07:35:34  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.12  2000/04/27 23:18:48  sparker
// Added problem initialization for MPM
//
// Revision 1.11  2000/04/26 06:48:33  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/24 15:17:01  sparker
// Fixed unresolved symbols
//
// Revision 1.9  2000/04/20 22:58:18  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.8  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.7  2000/04/19 21:20:03  dav
// more MPI stuff
//
// Revision 1.6  2000/04/19 05:26:11  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/13 06:50:57  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/11 07:10:40  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
