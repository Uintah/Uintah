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

using namespace Uintah;

OnDemandDataWarehouse::OnDemandDataWarehouse( const ProcessorGroup* myworld,
					      int generation, DataWarehouseP& parent) :
  d_lock("DataWarehouse lock"),
  DataWarehouse( myworld, generation, parent),
  d_responseTag( 0 )
{
  d_finalized = false;

}

void
OnDemandDataWarehouse::setGrid(const GridP& grid)
{
  d_grid = grid;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{

  for (reductionDBtype::const_iterator iter = d_reductionDB.begin(); 
       iter != d_reductionDB.end(); iter++) {
    delete iter->second->var;
    delete iter->second;
  }
  d_reductionDB.clear();

  for (dataLocationDBtype::const_iterator iter = d_dataLocation.begin();
       iter != d_dataLocation.end(); iter++) {
    for (int i = 0; i<iter->second->size(); i++ )
      delete &(iter->second[i]);
    delete iter->second;
  }
  d_dataLocation.clear();

  for (std::map<pair<const Patch*, const Patch*>, ScatterGatherBase* >::const_iterator iter = d_sgDB.begin(); iter != d_sgDB.end(); iter++) 
    delete iter->second;
  d_sgDB.clear();
#if 0
  for (psetDBType::const_iterator iter = d_psetDB.begin();
       iter != d_psetDB.end(); iter++) 
    delete iter->second;
  d_psetDB.clear();
#endif

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

   if(iter == d_reductionDB.end())
      throw UnknownVariable(label->getName(), "on reduction");

   var.copyPointer(*iter->second->var);
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, int matlIndex,
			      const Patch* patch)
{

   if(d_perpatchDB.exists(label, matlIndex, patch))
      return true;
   if(d_ncDB.exists(label, matlIndex, patch))
      return true;
   if(d_ccDB.exists(label, matlIndex, patch))
      return true;
   if(d_particleDB.exists(label, matlIndex, patch))
      return true;
   if(d_sfcxDB.exists(label,matlIndex,patch))
      return true;
   if(d_sfcyDB.exists(label,matlIndex,patch))
      return true;
   if(d_sfczDB.exists(label,matlIndex,patch))
      return true;
   return false;

}

void
OnDemandDataWarehouse::sendMPI(const VarLabel* label, int matlIndex,
			       const Patch* patch, const ProcessorGroup* world,
			       int dest, int tag, MPI_Request* requestid)
{
   if(d_ncDB.exists(label, matlIndex, patch)){
      NCVariableBase* var = d_ncDB.get(label, matlIndex, patch);
      void* buf;
      int count;
      MPI_Datatype datatype;
      var->getMPIBuffer(buf, count, datatype);
      //cerr << "ISend NC: buf=" << buf << ", count=" << count << ", dest=" << dest << ", tag=" << tag << ", comm=" << world->getComm() << ", req=" << requestid << '\n';
      MPI_Isend(buf, count, datatype, dest, tag, world->getComm(), requestid);
      return;
   }
   if(d_particleDB.exists(label, matlIndex, patch)){
      ParticleVariableBase* var = d_particleDB.get(label, matlIndex, patch);
      void* buf;
      int count;
      MPI_Datatype datatype;
      var->getMPIBuffer(buf, count, datatype);
      MPI_Isend(buf, count, datatype, dest, tag, world->getComm(), requestid);
      //cerr << "ISend Particle: buf=" << buf << ", count=" << count << ", dest=" << dest << ", tag=" << tag << ", comm=" << world->getComm() << ", req=" << requestid << '\n';
      return;
   }
   if(d_ccDB.exists(label, matlIndex, patch)) {
      throw InternalError("sendMPI not implemented for CC");
   }
   if(label->typeDescription()->getType() == TypeDescription::ScatterGatherVariable){
      throw InternalError("Sending sgvar shouldn't occur\n");
   }
   cerr << "Particles:\n";
   d_particleDB.print(cerr);
   cerr << "NC:\n";
   d_ncDB.print(cerr);
   cerr << "\n\n";
   throw UnknownVariable(label->getName(), patch->getID(), patch->toString(),
			 matlIndex, "in sendMPI");
}

void
OnDemandDataWarehouse::recvMPI(DataWarehouseP& old_dw,
			       const VarLabel* label, int matlIndex,
			       const Patch* patch, const ProcessorGroup* world,
			       int src, int tag, MPI_Request* requestid)
{
   switch(label->typeDescription()->getType()){
   case TypeDescription::ParticleVariable:
      {
	 if(d_particleDB.exists(label, matlIndex, patch))
	    throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
	 
	 // First, get the particle set.  We should already have it
	 ParticleSubset* pset = old_dw->getParticleSubset(matlIndex, patch);

	 Variable* v = label->typeDescription()->createInstance();
	 ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(v);
	 ASSERT(var != 0);
	 var->allocate(pset);
	 void* buf;
	 int count;
	 MPI_Datatype datatype;
	 var->getMPIBuffer(buf, count, datatype);
	 MPI_Irecv(buf, count, datatype, src, tag, world->getComm(), requestid);
	 d_particleDB.put(label, matlIndex, patch, var, false);
      }
   break;
   case TypeDescription::NCVariable:
      {
	 if(d_ncDB.exists(label, matlIndex, patch))
	    throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
	 Variable* v = label->typeDescription()->createInstance();
	 NCVariableBase* var = dynamic_cast<NCVariableBase*>(v);
	 ASSERT(var != 0);
	 var->allocate(patch->getNodeLowIndex(), patch->getNodeHighIndex());

	 void* buf;
	 int count;
	 MPI_Datatype datatype;
	 var->getMPIBuffer(buf, count, datatype);
	 MPI_Irecv(buf, count, datatype, src, tag, world->getComm(), requestid);
	 d_ncDB.put(label, matlIndex, patch, var, false);	 
      }
   break;
   case TypeDescription::ScatterGatherVariable:
      {
	 cerr << "recv sgvar notdone\n";
      }
   break;
   default:
      throw InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch));
   }
}

void
OnDemandDataWarehouse::reduceMPI(const VarLabel* label,
				 const ProcessorGroup* world)
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);

   if(iter == d_reductionDB.end())
      throw UnknownVariable(label->getName(), "on reduceMPI");

   void* sendbuf;
   int sendcount;
   MPI_Datatype senddatatype;
   MPI_Op sendop;
   iter->second->var->getMPIBuffer(sendbuf, sendcount, senddatatype, sendop);
   ReductionVariableBase* tmp = iter->second->var->clone();
   void* recvbuf;
   int recvcount;
   MPI_Datatype recvdatatype;
   MPI_Op recvop;
   tmp->getMPIBuffer(recvbuf, recvcount, recvdatatype, recvop);
   ASSERTEQ(recvcount, sendcount);
   ASSERTEQ(senddatatype, recvdatatype);
   ASSERTEQ(recvop, sendop);
      
   MPI_Allreduce(sendbuf, recvbuf, recvcount, recvdatatype, recvop,
		 world->getComm());
   iter->second->var->copyPointer(*tmp);
   delete tmp;
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

  char       * buffer = scinew char[ DWMpiHandler::MAX_BUFFER_SIZE ];
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
   ParticleSet* pset = scinew ParticleSet(numParticles);
   ParticleSubset* psubset = scinew ParticleSubset(pset, true, matlIndex, patch);
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
      throw UnknownVariable("ParticleSet", patch->getID(), patch->toString(),
			    matlIndex, "Cannot find particle set on patch");
   }
   return iter->second;
}

bool
OnDemandDataWarehouse::haveParticleSubset(int matlIndex, const Patch* patch)
{
   psetDBType::key_type key(matlIndex, patch);
   psetDBType::iterator iter = d_psetDB.find(key);
   return !(iter == d_psetDB.end());
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
   for (int i = 0; i<subsets.size(); i++ ) {
     delete subsets[i];
   }


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
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
      d_particleDB.get(label, matlIndex, patch, var);
   } else {
      const vector<const Patch*>& neighbors = pset->getNeighbors();
      const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();
      vector<ParticleVariableBase*> neighborvars(neighbors.size());
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor=neighbors[i];
	 if(!d_particleDB.exists(label, matlIndex, neighbors[i]))
	    throw UnknownVariable(label->getName(), neighbor->getID(),
				  neighbor->toString(), matlIndex,
				  neighbor == patch?"on patch":"on neighbor");
	 neighborvars[i] = d_particleDB.get(label, matlIndex, neighbors[i]);
      }
      var.gather(pset, neighbor_subsets, neighborvars);
   }
}

ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
					   ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getGhostType() == Ghost::None){
      if(!d_particleDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
      return d_particleDB.get(label, matlIndex, patch);
   } else {
      throw InternalError("getParticleVariable should not be used with ghost cells");
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
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
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
	       throw UnknownVariable(label->getName(), neighbor->getID(),
				     neighbor->toString(), matlIndex,
				     neighbor == patch?"on patch":"on neighbor");
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
     throw UnknownVariable(label->getName(), patch->getID(), patch->toString(),
			   matlIndex, "perpatch data");
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
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
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
	 lowIndex = patch->getGhostCellLowIndex(numGhostCells);
         highIndex = patch->getGhostCellHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalCells=0;
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low = lowIndex;
      IntVector high = highIndex;
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
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
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


void
OnDemandDataWarehouse::get(SFCXVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 1
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_sfcxDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
      d_sfcxDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCX Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCXLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCXHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalCells=0;
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high = patch->getGhostCellHighIndex(numGhostCells);
      level->selectPatches(low, high, neighbors);
      // modify it to only ignore corner nodes
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfcxDB.exists(label, matlIndex, neighbor))
	       throw InternalError("position variable does not exist: "+ 
				   label->getName());
	    SFCXVariableBase* srcvar = 
	       d_sfcxDB.get(label, matlIndex, neighbor);

	    using SCICore::Geometry::Max;
	    using SCICore::Geometry::Min;

	    IntVector low = Max(lowIndex, neighbor->getSFCXLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCXHighIndex());

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
OnDemandDataWarehouse::allocate(SFCXVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_sfcxDB.exists(label, matlIndex, patch))
      throw InternalError("SFCX variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCXLowIndex(), patch->getSFCXHighIndex());
}

void
OnDemandDataWarehouse::put(const SFCXVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_sfcxDB.exists(label, matlIndex, patch))
      throw InternalError("SFCX variable already exists: "+label->getName());

   // Put it in the database
   d_sfcxDB.put(label, matlIndex, patch, var.clone(), true);
}

void
OnDemandDataWarehouse::get(SFCYVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 1
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_sfcyDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
      d_sfcyDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCY Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCYLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCYHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalCells=0;
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high = patch->getGhostCellHighIndex(numGhostCells);
      level->selectPatches(low, high, neighbors);
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfcyDB.exists(label, matlIndex, neighbor))
	       throw InternalError("position variable does not exist: "+ 
				   label->getName());
	    SFCYVariableBase* srcvar = 
	       d_sfcyDB.get(label, matlIndex, neighbor);

	    using SCICore::Geometry::Max;
	    using SCICore::Geometry::Min;

	    IntVector low = Max(lowIndex, neighbor->getSFCYLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCYHighIndex());

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
OnDemandDataWarehouse::allocate(SFCYVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_sfcyDB.exists(label, matlIndex, patch))
      throw InternalError("SFCY variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCYLowIndex(), patch->getSFCYHighIndex());
}

void
OnDemandDataWarehouse::put(const SFCYVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_sfcyDB.exists(label, matlIndex, patch))
      throw InternalError("SFCY variable already exists: "+label->getName());

   // Put it in the database
   d_sfcyDB.put(label, matlIndex, patch, var.clone(), true);
}

void
OnDemandDataWarehouse::get(SFCZVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
#if 1
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
#endif
      if(!d_sfczDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch->getID(),
			       patch->toString(), matlIndex);
      d_sfczDB.get(label, matlIndex, patch, var);
#if 1
   } else {
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCZ Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCZLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCZHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalCells=0;
      const Level* level = patch->getLevel();
      std::vector<const Patch*> neighbors;
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high=patch->getGhostCellHighIndex(numGhostCells);
      level->selectPatches(low, high, neighbors);
      for(int i=0;i<neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfczDB.exists(label, matlIndex, neighbor))
	       throw InternalError("position variable does not exist: "+ 
				   label->getName());
	    SFCZVariableBase* srcvar = 
	       d_sfczDB.get(label, matlIndex, neighbor);

	    using SCICore::Geometry::Max;
	    using SCICore::Geometry::Min;

	    IntVector low = Max(lowIndex, neighbor->getSFCZLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCZHighIndex());

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
OnDemandDataWarehouse::allocate(SFCZVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
   // Error checking
   if(d_sfczDB.exists(label, matlIndex, patch))
      throw InternalError("SFCZ variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCZLowIndex(), patch->getSFCZHighIndex());
}

void
OnDemandDataWarehouse::put(const SFCZVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   ASSERT(!d_finalized);

   // Error checking
   if(d_sfczDB.exists(label, matlIndex, patch))
      throw InternalError("SFCZ variable already exists: "+label->getName());

   // Put it in the database
   d_sfczDB.put(label, matlIndex, patch, var.clone(), true);
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
      if(d_sfcxDB.exists(label, patch))
	 return true;
      if(d_sfcyDB.exists(label, patch))
	 return true;
      if(d_sfczDB.exists(label, patch))
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
   if(d_sfcxDB.exists(label, matlIndex, patch)) {
      SFCXVariableBase* var = d_sfcxDB.get(label, matlIndex, patch);
      var->emit(oc);
      return;
   }
   if(d_sfcyDB.exists(label, matlIndex, patch)) {
      SFCYVariableBase* var = d_sfcyDB.get(label, matlIndex, patch);
      var->emit(oc);
      return;
   }
   if(d_sfczDB.exists(label, matlIndex, patch)) {
      SFCZVariableBase* var = d_sfczDB.get(label, matlIndex, patch);
      var->emit(oc);
      return;
   }
   if(d_fcDB.exists(label, matlIndex, patch)) {
     FCVariableBase* var = d_fcDB.get(label, matlIndex, patch);
     var->emit(oc);
     return;
   }
   throw UnknownVariable(label->getName(), patch->getID(), patch->toString(),
			 matlIndex, "on emit");
}

void OnDemandDataWarehouse::emit(ostream& intout, const VarLabel* label) const
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter == d_reductionDB.end()){
      throw UnknownVariable(label->getName(), "on emit reduction");
   } else {
      iter->second->var->emit(intout);
   }
}

OnDemandDataWarehouse::ReductionRecord::ReductionRecord(ReductionVariableBase* var)
   : var(var)
{
}

void
OnDemandDataWarehouse::scatter(ScatterGatherBase* var, const Patch* from,
			       const Patch* to)
{
   pair<const Patch*, const Patch*> idx(from, to);
   if(d_sgDB.find(idx) != d_sgDB.end())
      throw InternalError("scatter variable already exists");
   d_sgDB[idx]=var;
}

ScatterGatherBase*
OnDemandDataWarehouse::gather(const Patch* from, const Patch* to)
{
   pair<const Patch*, const Patch*> idx(from, to);
   map<pair<const Patch*, const Patch*>, ScatterGatherBase*>::iterator iter
       = d_sgDB.find(idx);
   if(iter == d_sgDB.end())
      throw UnknownVariable("scatter/gather", from->getID(), from->toString(),
			   -1, " to patch "+to->toString());
   return iter->second;
}

void
OnDemandDataWarehouse::deleteParticles(ParticleSubset* delset)
{

}

//
// $Log$
// Revision 1.43  2000/08/08 01:32:45  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.42  2000/07/28 03:01:54  rawat
// modified createDatawarehouse and added getTop()
//
// Revision 1.41  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.40  2000/06/27 23:20:03  rawat
// added functions to deal with staggered cell variables. Also modified get function
// for CCVariables.
//
// Revision 1.39  2000/06/21 20:50:03  guilkey
// Added deleteParticles, a currently empty function that will remove
// particles that are no longer relevant to the simulation.
//
// Revision 1.38  2000/06/19 22:36:50  sparker
// Improved message for UnknownVariable
//
// Revision 1.37  2000/06/17 07:26:51  sparker
// Fixed a memory leak in scatter/gather
//
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
