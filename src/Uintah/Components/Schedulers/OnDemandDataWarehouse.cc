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
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Parallel/ProcessorGroup.h>
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

static const TypeDescription* specialType;

namespace Uintah {

OnDemandDataWarehouse::OnDemandDataWarehouse( const ProcessorGroup* myworld,
					      int generation ) :
  d_lock("DataWarehouse lock"),
  DataWarehouse( myworld, generation ),
  d_responseTag( 0 )
{
  d_finalized = false;
#if 0
  d_positionLabel = scinew VarLabel("__internal datawarehouse position variable",
				ParticleVariable<Point>::getTypeDescription(),
				VarLabel::Internal);
#endif

  if(!specialType)
     specialType = new TypeDescription(TypeDescription::Other, "DataWarehouse::specialInternalScatterGatherType", false);
  scatterGatherVariable = new VarLabel("DataWarehouse::scatterGatherVariable",
				       specialType, VarLabel::Internal);

  reloc_old_posLabel = reloc_new_posLabel = 0;
}

void
OnDemandDataWarehouse::setGrid(const GridP& grid)
{
  d_grid = grid;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
}

bool OnDemandDataWarehouse::isFinalized() const
{
   return d_finalized;
}

void OnDemandDataWarehouse::finalize()
{
   d_finalized=true;
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

bool
OnDemandDataWarehouse::exists(const VarLabel* label, int matlIndex,
			      const Patch* patch)
{

  if(d_perpatchDB.exists(label,matlIndex,patch))
	return true;
  if(d_ncDB.exists(label,matlIndex,patch))
	return true;
  if(d_ccDB.exists(label,matlIndex,patch))
	return true;

  return false;

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
   } else {
      iter->second->var->reduce(var);
   }
}

void
OnDemandDataWarehouse::override(const ReductionVariableBase& var,
				const VarLabel* label)
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter != d_reductionDB.end())
      delete d_reductionDB[label];
   d_reductionDB[label]=scinew ReductionRecord(var.clone());
}

void
OnDemandDataWarehouse::sendMpiDataRequest( const string & /*varName*/,
					         Patch * /*patch*/,
					         int      /*numGhostCells*/ )
{
  if( d_myworld->size() == 1 ) {
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

  request.fromMpiRank = d_myworld->myrank();
  request.toMpiRank = !d_myworld->myrank(); // Just a testing hack...

  d_lock.writeLock();
  request.tag = d_responseTag;
  currentTag = d_responseTag;
  d_responseTag++;
  d_lock.writeUnlock();

  request.type = DWMpiHandler::GridVar;
  sprintf( request.varName, "varA" );
  request.patch = 0;
  request.generation = d_generation;

  cerr << "OnDemandDataWarehouse " << d_myworld->myrank() << ": sending data request\n";

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

ParticleSubset*
OnDemandDataWarehouse::createParticleSubset(particleIndex numParticles,
					    int matlIndex, const Patch* patch)
{
   ParticleSet* pset = new ParticleSet(numParticles);
   ParticleSubset* psubset = new ParticleSubset(pset, true, matlIndex, patch);
   psetDBType::key_type key(matlIndex, patch);
   if(d_psetDB.find(key) != d_psetDB.end())
      throw InternalError("createParticleSubset called twice for patch");
   d_psetDB[key]=psubset;
   return psubset;
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch)
{
   psetDBType::key_type key(matlIndex, patch);
   psetDBType::iterator iter = d_psetDB.find(key);
   if(iter == d_psetDB.end()){
      cerr << "matlIndex = " << matlIndex << '\n';
      cerr << "patch=" << patch << '\n';
      throw UnknownVariable("Cannot find particle set on patch");
   }
   return iter->second;
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch,
					 Ghost::GhostType gtype,
					 int numGhostCells,
					 const VarLabel* pos_var)
{
   if(gtype == Ghost::None){
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      return getParticleSubset(matlIndex, patch);
   }
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
   Box box = patch->getGhostBox(IntVector(l,l,l), IntVector(h,h,h));
   particleIndex totalParticles = 0;
   vector<ParticleVariableBase*> neighborvars;
   vector<ParticleSubset*> subsets;
   const Level* level = patch->getLevel();
   std::vector<const Patch*> neighbors;
   IntVector low(patch->getCellLowIndex()+IntVector(l,l,l));
   IntVector high(patch->getCellHighIndex()+IntVector(h,h,h));
   level->selectPatches(low, high, neighbors);
   for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor){
	 ParticleSubset* pset = getParticleSubset(matlIndex, neighbor);
	 ParticleVariable<Point> pos;
	 get(pos, pos_var, pset);
	 ParticleSubset* subset = 
	    scinew ParticleSubset(pset->getParticleSet(), false, -1, 0);
	 for(ParticleSubset::iterator iter = pset->begin();
	     iter != pset->end(); iter++){
	    particleIndex idx = *iter;
	    if(box.contains(pos[idx]))
	       subset->addParticle(idx);
	 }
	 totalParticles+=subset->numParticles();
	 subsets.push_back(subset);
      }
   }
   ParticleSet* newset = scinew ParticleSet(totalParticles);
   ParticleSubset* newsubset = scinew ParticleSubset(newset, true,
						     matlIndex, patch,
						     gtype, numGhostCells,
						     neighbors, subsets);
   return newsubset;
}

void
OnDemandDataWarehouse::get(ParticleVariableBase& var,
			   const VarLabel* label,
			   ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getGhostType() == Ghost::None){
      if(!d_particleDB.exists(label, matlIndex, patch))
	 throw UnknownVariable("Unknown variable on neighbor: "+label->getName());
      d_particleDB.get(label, matlIndex, patch, var);
   } else {
      const vector<const Patch*>& neighbors = pset->getNeighbors();
      const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();
      vector<ParticleVariableBase*> neighborvars(neighbors.size());
      for(int i=0;i<neighbors.size();i++){
	 if(!d_particleDB.exists(label, matlIndex, neighbors[i]))
	    throw UnknownVariable("Unknown variable on neighbor: "+label->getName());
	 neighborvars[i] = d_particleDB.get(label, matlIndex, neighbors[i]);
      }
      var.gather(pset, neighbor_subsets, neighborvars);
   }
}

void
OnDemandDataWarehouse::allocate(ParticleVariableBase& var,
				const VarLabel* label,
				ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Particle variable already exists: " +
			  label->getName());

   var.allocate(pset);
}

void
OnDemandDataWarehouse::put(const ParticleVariableBase& var,
			   const VarLabel* label)
{
   ASSERT(!d_finalized);

   ParticleSubset* pset = var.getParticleSubset();
   if(pset->numGhostCells() != 0 || pset->getGhostType() != 0)
      throw InternalError("ParticleVariable cannot use put with ghost cells");
   const Patch* patch = pset->getPatch();
   int matlIndex = pset->getMatlIndex();

   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Variable already exists: "+label->getName());

   // Put it in the database
   d_particleDB.put(label, matlIndex, patch, var.clone(), true);
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
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low(patch->getCellLowIndex()+IntVector(l,l,l));
      IntVector high(patch->getCellHighIndex()+IntVector(h,h,h));
      level->selectPatches(low, high, neighbors);
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
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
   d_ncDB.put(label, matlIndex, patch, var.clone(), true);
}

void
OnDemandDataWarehouse::get(PerPatchBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch)
{
  if(!d_perpatchDB.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName());
  d_perpatchDB.get(label, matlIndex, patch, var);

}

void
OnDemandDataWarehouse::put(const PerPatchBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_perpatchDB.exists(label, matlIndex, patch))
     throw InternalError("PerPatch variable already exists: "+label->getName());

   // Put it in the database
   d_perpatchDB.put(label, matlIndex, patch, var.clone(), true);
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
#if 0
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_ccDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName());
      d_ccDB.get(label, matlIndex, patch, var);
#if 0
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
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low(patch->getCellLowIndex()+IntVector(l,l,l));
      IntVector high(patch->getCellHighIndex()+IntVector(h,h,h));
      level->selectPatches(low, high, neighbors);
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
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
   d_ccDB.put(label, matlIndex, patch, var.clone(), true);
}

void
OnDemandDataWarehouse::allocate(FCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_fcDB.exists(label, matlIndex, patch))
      throw InternalError("FC variable already exists: "+label->getName());

   // Allocate the variable
   // Probably should be getFaceLowIndex() . . .
   var.allocate(patch->getCellLowIndex(), patch->getCellHighIndex());
}

void
OnDemandDataWarehouse::get(FCVariableBase& var, const VarLabel* label,
			   int matlIndex,
			   const Patch* patch, Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 0
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_fcDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName());
      d_fcDB.get(label, matlIndex, patch, var);
#if 0
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
		  if(!d_fcDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
					 label->getName());
		  FCVariableBase* srcvar = 
		    d_fcDB.get(label, matlIndex, neighbor);

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
OnDemandDataWarehouse::put(const FCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch )
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_fcDB.exists(label, matlIndex, patch))
      throw InternalError("FC variable already exists: "+label->getName());

   // Put it in the database
   d_fcDB.put(label, matlIndex, patch, var.clone(), true);
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

#if 0
  variableListType::iterator iter = varList->begin();

  // Commented out because "contains" got axed, and it probably
  // isn't the right thing to do - Steve
  while( iter != varList->end() ) {
    if( (*iter)->patch->contains( *patch ) ) {
      return (*iter)->mpiNode;
    }
    iter++;
  }
#endif

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
      if(d_ccDB.exists(label, patch))
	 return true;
      if(d_fcDB.exists(label, patch))
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
   if(d_ccDB.exists(label, matlIndex, patch)) {
     CCVariableBase* var = d_ccDB.get(label, matlIndex, patch);
     var->emit(oc);
     return;
   }
   if(d_fcDB.exists(label, matlIndex, patch)) {
     FCVariableBase* var = d_fcDB.get(label, matlIndex, patch);
     var->emit(oc);
     return;
   }
   throw UnknownVariable(label->getName());
}

void OnDemandDataWarehouse::emit(ostream& intout, const VarLabel* label) const
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter == d_reductionDB.end()){
      throw UnknownVariable(label->getName());
   } else {
      iter->second->var->emit(intout);
   }
}

OnDemandDataWarehouse::ReductionRecord::ReductionRecord(ReductionVariableBase* var)
   : var(var)
{
}

struct ScatterMaterialRecord {
   ParticleSubset* relocset;
   vector<ParticleVariableBase*> vars;
};

struct ScatterRecord {
   vector<ScatterMaterialRecord*> matls;
};

void
OnDemandDataWarehouse::scatterParticles(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
   ASSERT(new_dw.get_rep() == this);

   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   vector<const Patch*> neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr(neighbors.size());
   for(int i=0;i<sr.size();i++)
      sr[i]=0;
   for(int m = 0; m < reloc_numMatls; m++){
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      get(px, reloc_old_posLabel, pset);

      ParticleSubset* relocset = new ParticleSubset(pset->getParticleSet(),
						    false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(!patch->getBox().contains(px[idx])){
	    //cerr << "WARNING: Particle left patch: " << px[idx] << ", patch: " << patch << '\n';
	    relocset->addParticle(idx);
	 }
      }
      if(relocset->numParticles() > 0){
	 // Figure out where they went...
	 for(ParticleSubset::iterator iter = relocset->begin();
	     iter != relocset->end(); iter++){
	    particleIndex idx = *iter;
	    // This loop should change - linear searches are not good!
	    int i;
	    for(i=0;i<neighbors.size();i++){
	       if(neighbors[i]->getBox().contains(px[idx])){
		  break;
	       }
	    }
	    if(i == neighbors.size()){
	       // Make sure that the particle left the world
	       if(level->containsPoint(px[idx]))
		  throw InternalError("Particle fell through the cracks!");
	    } else {
	       if(!sr[i]){
		  sr[i] = new ScatterRecord();
		  sr[i]->matls.resize(reloc_numMatls);
		  for(int m=0;m<reloc_numMatls;m++){
		     sr[i]->matls[m]=0;
		  }
	       }
	       if(!sr[i]->matls[m]){
		  ScatterMaterialRecord* smr=new ScatterMaterialRecord();
		  sr[i]->matls[m]=smr;
		  smr->vars.push_back(d_particleDB.get(reloc_old_posLabel, m, patch));
		  for(int v=0;v<reloc_old_labels.size();v++)
		     smr->vars.push_back(d_particleDB.get(reloc_old_labels[v], m, patch));
		  smr->relocset = new ParticleSubset(pset->getParticleSet(),
						     false, -1, 0);
	       }
	       sr[i]->matls[m]->relocset->addParticle(idx);
	    }
	 }
      } else {
	 delete relocset;
      }
   }
   for(int i=0;i<sr.size();i++){
      if(patch != neighbors[i]){
	 pair<const Patch*, const Patch*> idx(patch, neighbors[i]);
	 if(d_sgDB.find(idx) != d_sgDB.end())
	    throw InternalError("Scatter/Gather Variable duplicated?");
	 d_sgDB[idx] = sr[i];
      } else {
	 if(sr[i])
	    throw InternalError("Patch scattered particles to itself?");
      }
   }
}

void
OnDemandDataWarehouse::gatherParticles(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
   ASSERT(new_dw.get_rep() == this);

   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   vector<const Patch*> neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr;
   for(int i=0;i<neighbors.size();i++){
      pair<const Patch*, const Patch*> idx(neighbors[i], patch);
      if(patch != neighbors[i]){
	 map<pair<const Patch*, const Patch*>, ScatterRecord*>::iterator iter = d_sgDB.find(idx);
	 if(iter == d_sgDB.end())
	    throw InternalError("Did not receive a scatter?");
	 if(iter->second)
	    sr.push_back(iter->second);
      }
   }
   for(int m=0;m<reloc_numMatls;m++){
      // Compute the new particle subset
      vector<ParticleSubset*> subsets;
      vector<ParticleVariableBase*> posvars;

      // Get the local subset without the deleted particles...
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      get(px, reloc_old_posLabel, pset);

      ParticleSubset* keepset = new ParticleSubset(pset->getParticleSet(),
						   false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(patch->getBox().contains(px[idx]))
	    keepset->addParticle(idx);
      }
      subsets.push_back(keepset);
      particleIndex totalParticles = keepset->numParticles();
      ParticleVariableBase* pos = d_particleDB.get(reloc_old_posLabel, m, patch);
      posvars.push_back(pos);

      // Get the subsets from the neighbors
      for(int i=0;i<sr.size();i++){
	 if(sr[i]->matls[m]){
	    subsets.push_back(sr[i]->matls[m]->relocset);
	    posvars.push_back(sr[i]->matls[m]->vars[0]);
	    totalParticles += sr[i]->matls[m]->relocset->numParticles();
	 }
      }
      ParticleVariableBase* newpos = pos->clone();
      ParticleSet* newset = new ParticleSet(totalParticles);
      ParticleSubset* newsubset = new ParticleSubset(newset, true, m, patch);
      newpos->gather(newsubset, subsets, posvars);
      if(d_particleDB.exists(reloc_new_posLabel, m, patch))
	 throw InternalError("Variable already exists: "+reloc_new_posLabel->getName());
      d_particleDB.put(reloc_new_posLabel, m, patch, newpos, false);

      for(int v=0;v<reloc_old_labels.size();v++){
	 vector<ParticleVariableBase*> gathervars;
	 ParticleVariableBase* var = d_particleDB.get(reloc_old_labels[v],
						      m, patch);
	 gathervars.push_back(var);
	 for(int i=0;i<sr.size();i++){
	    if(sr[i]->matls[m])
	       gathervars.push_back(sr[i]->matls[m]->vars[v+1]);
	 }
	 ParticleVariableBase* newvar = var->clone();
	 newvar->gather(newsubset, subsets, gathervars);
	 if(d_particleDB.exists(reloc_new_labels[v], m, patch))
	    throw InternalError("Variable already exists: "+reloc_new_labels[v]->getName());
	 d_particleDB.put(reloc_new_labels[v], m, patch, newvar, false);
      }

      psetDBType::key_type key(m, patch);
      if(d_psetDB.find(key) != d_psetDB.end())
	 throw InternalError("ParticleSet already exists on patch");
      d_psetDB[key]=newsubset;
   }
}

void
OnDemandDataWarehouse::scheduleParticleRelocation(const LevelP& level,
						  SchedulerP& sched,
						  DataWarehouseP& old_dw,
						  const VarLabel* old_posLabel,
						  const vector<const VarLabel*>& old_labels,
						  const VarLabel* new_posLabel,
						  const vector<const VarLabel*>& new_labels,
						  int numMatls)
{
   reloc_old_posLabel = old_posLabel;
   reloc_old_labels = old_labels;
   reloc_new_posLabel = new_posLabel;
   reloc_new_labels = new_labels;
   reloc_numMatls = numMatls;
   ASSERTEQ(reloc_new_labels.size(), reloc_old_labels.size());
   DataWarehouseP new_dw (this);
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;

      Task* t = scinew Task("OnDemandDataWarehouse::scatterParticles",
			    patch, old_dw, new_dw,
			    this, &OnDemandDataWarehouse::scatterParticles);
      for(int m=0;m < numMatls;m++){
	 t->requires( this, old_posLabel, m, patch, Ghost::None);
	 for(int i=0;i<old_labels.size();i++)
	    t->requires( this, old_labels[i], m, patch, Ghost::None);
      }
      t->computes(this, scatterGatherVariable, 0, patch);
      sched->addTask(t);

      Task* t2 = scinew Task("OnDemandDataWarehouse::gatherParticles",
			     patch, old_dw, new_dw,
			     this, &OnDemandDataWarehouse::gatherParticles);
      // Particles are only allowed to be one cell out
      IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
      IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
      std::vector<const Patch*> neighbors;
      level->selectPatches(l, h, neighbors);
      for(int i=0;i<neighbors.size();i++)
	 t2->requires(this, scatterGatherVariable, 0, neighbors[i], Ghost::None);
      for(int m=0;m < numMatls;m++){
	 t2->computes( this, new_posLabel, m, patch);
	 for(int i=0;i<new_labels.size();i++)
	    t2->computes(this, new_labels[i], m, patch);
      }

      sched->addTask(t2);
   }
}

} // end namespace Uintah

//
// $Log$
// Revision 1.36  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.35  2000/06/16 19:48:55  sparker
// Eliminated carryForward
//
// Revision 1.34  2000/06/16 05:03:07  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.33  2000/06/15 21:57:11  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.32  2000/06/14 23:39:26  jas
// Added FCVariables.
//
// Revision 1.31  2000/06/14 00:31:06  jas
// Added cc_DB to the emit method.
//
// Revision 1.30  2000/06/05 19:50:22  guilkey
// Added functionality for PerPatch variable.
//
// Revision 1.29  2000/06/03 05:27:23  sparker
// Fixed dependency analysis for reduction variables
// Removed warnings
// Now allow for task patch to be null
// Changed DataWarehouse emit code
//
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
