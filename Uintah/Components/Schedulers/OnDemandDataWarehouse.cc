/* REFERENCED */
static char *id="@(#) $Id$";

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Guard.h>
#include <SCICore/Geometry/Point.h>

#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Exceptions/UnknownVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Region.h>

#include <iostream>
#include <string>
#include <mpi.h>

using SCICore::Exceptions::InternalError;
using SCICore::Thread::Guard;
using SCICore::Geometry::Point;
using std::cerr;
using std::string;
using std::vector;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
using namespace Uintah;

OnDemandDataWarehouse::OnDemandDataWarehouse( int MpiRank, int MpiProcesses )
  : d_lock("DataWarehouse lock"), DataWarehouse( MpiRank, MpiProcesses )
{
  d_finalized = false;
  d_positionLabel = new VarLabel("__internal datawarehouse position variable",
				ParticleVariable<Point>::getTypeDescription(),
				VarLabel::Internal);

  if( MpiProcesses != 1 ) {
    d_worker = new DataWarehouseMpiHandler( this );
    d_thread = new Thread( d_worker, "DataWarehouseMpiHandler" );
  } else {
    // If there is only one MPI process, then their is only one
    // DataWarehouse, so we don't need to use MPI to transfer any
    // data.
    d_worker = 0;
    d_thread = 0;
  }
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
			   const VarLabel* label) const
{
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter == d_reductionDB.end())
      throw UnknownVariable(label->getName());

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
   reductionDBtype::const_iterator iter = d_reductionDB.find(label);
   if(iter == d_reductionDB.end()){
      d_reductionDB[label]=new ReductionRecord(var.clone());
      iter = d_reductionDB.find(label);
   } else {
      iter->second->var->reduce(var);
   }
}

void
OnDemandDataWarehouse::get(ParticleVariableBase& var,
			   const VarLabel* label,
			   int matlIndex,
			   const Region* region,
			   Ghost::GhostType gtype,
			   int numGhostCells) const
{
   if(!d_particleDB.exists(label, matlIndex, region))
      throw UnknownVariable(label->getName());
   if(gtype == Ghost::None){
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      d_particleDB.get(label, matlIndex, region, var);
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
      Box box = region->getGhostBox(l, h);
      particleIndex totalParticles = 0;
      vector<ParticleVariableBase*> neighborvars;
      vector<ParticleSubset*> subsets;
      for(int ix=l;ix<=h;ix++){
	 for(int iy=l;iy<=h;iy++){
	    for(int iz=l;iz<=h;iz++){
	       const Region* neighbor = region->getNeighbor(IntVector(ix,iy,iz));
	       if(neighbor){
		  if(!d_particleDB.exists(d_positionLabel, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
			  d_positionLabel->getName());
		  if(!d_particleDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
			  d_positionLabel->getName());

		  ParticleVariable<Point> pos;
		  d_particleDB.get(d_positionLabel, matlIndex, neighbor, pos);
		  ParticleSubset* pset = pos.getParticleSubset();
		  ParticleSubset* subset = new ParticleSubset(pset->getParticleSet(), false);
		  for(ParticleSubset::iterator iter = pset->begin();
		      iter != pset->end(); iter++){
		     particleIndex idx = *iter;
		     if(box.contains(pos[idx])){
			subset->addParticle(idx);
		     }
		  }
		  totalParticles+=subset->numParticles();
		  neighborvars.push_back(d_particleDB.get(label, matlIndex, neighbor));
		  subsets.push_back(subset);
	       }
	    }
	 }
      }
      ParticleSet* newset = new ParticleSet(totalParticles);
      ParticleSubset* newsubset = new ParticleSubset(newset, true);
      var.gather(newsubset, subsets, neighborvars);
   }
}

void
OnDemandDataWarehouse::allocate(int numParticles,
				ParticleVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Region* region)
{
   // Error checking
   if(d_particleDB.exists(label, matlIndex, region))
      throw InternalError("Particle variable already exists: "+label->getName());
   if(!label->isPositionVariable())
      throw InternalError("Particle allocate via numParticles should "
			  "only be used for position variables");
   if(d_particleDB.exists(d_positionLabel, matlIndex, region))
      throw InternalError("Particle position already exists in datawarehouse");

   // Create the particle set and variable
   ParticleSet* pset = new ParticleSet(numParticles);
   ParticleSubset* psubset = new ParticleSubset(pset, true);
   var.allocate(psubset);

   // Put it in the database
   d_particleDB.put(d_positionLabel, matlIndex, region, var, false);
}

void
OnDemandDataWarehouse::allocate(ParticleVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Region* region)
{
   // Error checking
   if(d_particleDB.exists(label, matlIndex, region))
      throw InternalError("Particle variable already exists: " +
			  label->getName());
   if(!d_particleDB.exists(d_positionLabel, matlIndex, region))
      throw InternalError("Position variable does not exist: " + 
			  d_positionLabel->getName());
   ParticleVariable<Point> pos;
   d_particleDB.get(d_positionLabel, matlIndex, region, pos);

   // Allocate the variable
   var.allocate(pos.getParticleSubset());
}

void
OnDemandDataWarehouse::put(const ParticleVariableBase& var,
			   const VarLabel* label,
			   int matlIndex,
			   const Region* region)
{
   // Error checking
   if(d_particleDB.exists(label, matlIndex, region))
      throw InternalError("Variable already exists: "+label->getName());

   // Put it in the database
   d_particleDB.put(label, matlIndex, region, var, true);
}

void
OnDemandDataWarehouse::get(NCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Region* region,
			   Ghost::GhostType gtype,
			   int numGhostCells) const
{
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_ncDB.exists(label, matlIndex, region))
	 throw UnknownVariable(label->getName());
      d_ncDB.get(label, matlIndex, region, var);
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
	 lowIndex = region->getNodeLowIndex()-gc;
	 highIndex = region->getNodeHighIndex()+gc;
	 cerr << "Nodes around nodes is probably not functional!\n";
	 break;
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Uppwer neighbors
	 l=0;
	 h=1;
	 lowIndex = region->getCellLowIndex();
         highIndex = region->getCellHighIndex()+gc;
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      long totalNodes=0;
      for(int ix=l;ix<=h;ix++){
	 for(int iy=l;iy<=h;iy++){
	    for(int iz=l;iz<=h;iz++){
	       const Region* neighbor = region->getNeighbor(IntVector(ix,iy,iz));
	       if(neighbor){
		  if(!d_ncDB.exists(label, matlIndex, neighbor))
		     throw InternalError("Position variable does not exist: "+ 
					 label->getName());
		  NCVariableBase* srcvar = d_ncDB.get(label, matlIndex, neighbor);
		  IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
		  IntVector high = Min(highIndex, neighbor->getNodeHighIndex());
		  if(high.x() < low.x() || high.y() < low.y() || high.z() < low.z())
		     throw InternalError("Region doesn't overlap?");
		  var.copyRegion(srcvar, low, high);
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
}

void
OnDemandDataWarehouse::allocate(NCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Region* region)
{
   // Error checking
   if(d_ncDB.exists(label, matlIndex, region))
      throw InternalError("NC variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(region->getNodeLowIndex(), region->getNodeHighIndex());
}

void
OnDemandDataWarehouse::put(const NCVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Region* region)
{
   // Error checking
   if(d_ncDB.exists(label, matlIndex, region))
      throw InternalError("NC variable already exists: "+label->getName());

   // Put it in the database
   d_ncDB.put(label, matlIndex, region, var, true);
}

int
OnDemandDataWarehouse::findMpiNode( const VarLabel * label,
				    const Region   * region )
{
  variableListType * varList = d_dataLocation[ label ];

  char msg[ 1024 ];

  if( varList == 0 ) {
    sprintf( msg, "findMpiNode: Requested variable: %s for\n"
	     "region %s is not in d_dataLocation",
	     label->getName().c_str(), region->toString().c_str() );
    throw InternalError( string( msg ) );
  }
  // Run through all the different regions associated with "label" to
  // find which one contains the "region" that has been requested.

  variableListType::iterator iter = varList->begin();

  while( iter != varList->end() ) {
    if( (*iter)->region->contains( *region ) ) {
      return (*iter)->mpiNode;
    }
    iter++;
  }

  sprintf( msg, "findMpiNode: Requested region: %s for\n"
	   "region %s is not in d_dataLocation",
	   label->getName().c_str(), region->toString().c_str() );
  throw InternalError( string( msg ) );
}

void
OnDemandDataWarehouse::registerOwnership( const VarLabel * label,
					  const Region   * region,
					        int        mpiNode )
{
  variableListType * varList = d_dataLocation[ label ];

  if( varList == 0 ) {
    varList = new variableListType();
    d_dataLocation[ label ] = varList;
  }

  // Possibly should make sure that varList does not already have
  // this "label" with this "region" in it...  for now assuming that 
  // this doesn't happen...

  dataLocation * location = new dataLocation();

  location->region = region;
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
      for(Level::const_regionIterator iter = level->regionsBegin();
	  iter != level->regionsEnd(); iter++){
	 const Region* region = *iter;

	 d_particleDB.copyAll(from->d_particleDB, d_positionLabel, region);
      }
   }
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, const Region* region) const
{
   if(!region){
      reductionDBtype::const_iterator iter = d_reductionDB.find(label);
      if(iter != d_reductionDB.end())
	 return true;
   } else {
      if(d_ncDB.exists(label, region))
	 return true;
      if(d_particleDB.exists(label, region))
	 return true;
   }
   return false;
}

OnDemandDataWarehouse::ReductionRecord::ReductionRecord(ReductionVariableBase* var)
   : var(var)
{
}

///////////////////////////////////////////////////////////////
// DataWarehouseMpiHandler Routines:

const int DataWarehouseMpiHandler::MAX_BUFFER_SIZE = 1024;
const int DataWarehouseMpiHandler::MPI_DATA_REQUEST_TAG = 123321;

DataWarehouseMpiHandler::DataWarehouseMpiHandler( DataWarehouse * dw ) :
  d_dw( dw )
{
}

void
DataWarehouseMpiHandler::run()
{
  if( d_dw->d_MpiProcesses == 1 ) {
    throw InternalError( "DataWarehouseMpiHandler should not be running "
			 "if there is only one MPI process." );
  }

  MPI_Status status;
  char       buffer[ MAX_BUFFER_SIZE ];
  bool       done = false;

  while( !done ) {

    MPI_Recv( buffer, sizeof( MpiDataRequest ), MPI_BYTE, MPI_ANY_SOURCE,
	      MPI_DATA_REQUEST_TAG, MPI_COMM_WORLD, &status );

    MpiDataRequest * request = (MpiDataRequest *) buffer;

    cerr << "OnDemandDataWarehouse " << d_dw->d_MpiRank << " received a " 
	 << "request that " << status.MPI_SOURCE << " wants me to "
	 << "send it information of type: " << request->type << "\n";
    cerr << "   It wants data for variable: " << request->varName 
	 << " in retion of " << request->region << "\n";

    if( d_dw->d_MpiRank != request->toMpiRank || 
	status.MPI_SOURCE != request->fromMpiRank ) {
      throw InternalError( "Data Notification Message was corrupt" );
    }

    if( request->type == ReductionVar ) {
      cerr << "Received a reduction var... need to get it from my "
	   << "database\n";
    } else if( request->type == GridVar ) {
      cerr << "Received a grid var... need to get it from my "
	   << "database\n";
    } else {
      throw InternalError( "Do not know how to handle this type of data" );
    }

    // Look up the varName in the DW.  ??What to do if it is not there??

    // Pull data out of DataWarehouse and pack in into "buffer"

    // figure out how big the data is...
    int size = 1;

    MPI_Send( buffer, size, MPI_BYTE, status.MPI_SOURCE, request->tag,
	      MPI_COMM_WORLD );

  } // end while
}

//
// $Log$
// Revision 1.20  2000/05/10 20:02:53  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
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
